import importlib.util
reqs = ['marvin', 'watchdog', 'openai', 'retry']
if not all(importlib.util.find_spec(req) for req in reqs):
    import pip
    pip.main(['install'] + reqs)

import os
import sys
import re
import time
import logging
import subprocess
import tempfile
from typing import Optional
from marvin import ai_fn
import marvin
from retry import retry
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, PatternMatchingEventHandler
from pydantic import ValidationError
from openai.error import Timeout

marvin.settings.llm_model = "openai/gpt-4-1106-preview" # 4 Turbo
marvin.settings.llm_temperature = 0.2
marvin.settings.llm_request_timeout_seconds = 30

logger = None

def init_logging():
    global logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        # In case we reload this module from a notebook
        logger.handlers.clear()
    # file_handler = logging.FileHandler(f'{LOG_BASE_DIR}/tcr.log')
    # logger.addHandler(file_handler)

class StepFailedException(Exception):
    def __init__(self, message):
        self.message = message

def run_and_check_success(command: str, env):
    logger.info("running %s", command)
    proc = subprocess.run(command, shell=True, text=True, env=env, capture_output=True)
    if proc.returncode != 0:
        out = proc.stdout
        err = proc.stderr
        if out:
            logger.info("Out: %s", out)
        if err:
            logger.warn("Err: %s", err)
        raise StepFailedException(command) 
    else:
        logger.info("success")

def git_diff_head(staged=False) -> str:
    staged_arg = []
    if staged:
        staged_arg = ["--staged"]
    args = ["git", "diff"] + staged_arg + ["HEAD", "--", ".", ":!Pipfile.lock"]
    return subprocess.check_output(args, text=True)

def git_status() -> str:
    args = ["git", "status"]
    return subprocess.check_output(args, text=True)

def git_commit(message: str) -> None:
    """Invoke git commit, allowing user to edit"""
    args = ["git", "commit", "-am", message]
    subprocess.run(args, check=True)


def git_show_top_level() -> Optional[str]:
    try:
        args = ["git", "rev-parse", "--show-toplevel"]
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return None



@ai_fn
@retry((ValidationError, Timeout), tries=3, delay=1)
def suggest(intent: str, code: str) -> list[str]:
    """Suggest 2-5 incremental code improvements responsive to the provided intent"""

@ai_fn
@retry((ValidationError, Timeout), tries=3, delay=1)
def commit(diff: str) -> str:
    """
    Provide a one-line commit message that describes what happened in the diff.
    It is probably related to the intent, possibly related to the suggestions.
    """

def get_focused_code(focus):
    """
    Return the code in the focused file.
    Valid focus formats:
        <file>
        <file>:<line>
        <file>:<start_line>-<end_line>
    If a single line is selected, include 10 lines of context on either side.
    """
    parts = focus.split(':')
    file_name = parts[0]
    line_range_text = parts[1] if len(parts) > 1 else None
    with open(file_name) as f:
        lines = f.readlines()
        # range not supported yet
    if len(lines) == 0:
        return ""
    start_line = 0
    end_line = -1
    if line_range_text:
        line_range_parts = line_range_text.split('-')
        if len(line_range_parts) == 1:
            context_lines = 10
            mid_line = int(line_range_parts[0])
            start_line = max(0, mid_line - context_lines)
            end_line = min(len(lines), mid_line + context_lines)
        elif len(line_range_parts) == 2:
            start_line = int(line_range_parts[0])
            end_line = int(line_range_parts[1])
    return ''.join(lines[start_line:end_line])

LINE_RANGE_PATTERN = re.compile(r"\d+(-\d+)?")

def validate_focus(focus):
    """Ensure focus string would be accepted by get_focused_code()"""
    parts = focus.split(':')
    file_name = parts[0]
    line_range = parts[1] if len(parts) > 1 else None
    valid_file = os.path.isfile(file_name)
    
    valid_line_range = line_range is None or re.fullmatch(LINE_RANGE_PATTERN, line_range)
    return valid_file and valid_line_range

class MyFileSystemEventHandler(PatternMatchingEventHandler):
    def __init__(self, test_cmd, focus, intent, suggestions=None):
        self.test_cmd = test_cmd
        self.focus = focus
        self.intent = intent
        self.suggestions = suggestions or []
        super().__init__(ignore_patterns=['**/.git','**/.git/**'])

    def on_any_event(self, event):
        """
        After a short wait for concurrent changes, signal that code was updated 
        """
        time.sleep(0.3)
        self.code_updated()
    
    def code_updated(self):
        """
        * Check git diff for changes
        * Run tests, continue if passed
        * Prompt user, do one of:
          * Commit with GPT message
          * Revert changes
          * Change focus section
        * Show GPT suggestions on focus section
        """
        env = os.environ
        revert_step = 'git restore .'
        diff_text = git_diff_head()
        if diff_text.strip():
            print(git_status())
            try:
                run_and_check_success(self.test_cmd, env=env)
            except StepFailedException as e:
                # Could revert here? But might be halfway through a multi-file change.
                print(f"Tests failed...")
                return
            print(f"Test passed...")
            message = commit(diff=diff_text)
            print(f"Message: {message}")
            print(f"Focus on {self.focus}")
            print(f"y to accept, r to reset, f <path> to change focus")
            while (user_input := input()) not in 'yYrR':
                input_parts = user_input.split()
                if input_parts[0] == "f":
                    if validate_focus(input_parts[1]):
                        self.focus = input_parts[1]
                    else:
                        print("Invalid focus.")
            if user_input in 'yY':
                git_commit(message)
            elif user_input in 'rR':
                run_and_check_success(revert_step, env=env)
            code = get_focused_code(self.focus)
            self.suggestions = suggest(intent=self.intent, code=code)
            print("Suggestions:")
            print("\n".join(f'  * {s}' for s in self.suggestions))
            print("Watching for changes...")
            


def run_suggest_loop(watch_path='.'):
    """
    Watch files for changes.
    On change: test, prompt for commit, make suggestions.
    """
    intent = "Improve this code, preferably small automated refactoring steps"
    event_handler = MyFileSystemEventHandler(
        focus='hello.py', 
        test_cmd="just test", 
        intent=intent)
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=True)
    # In case it was already updated
    event_handler.code_updated()
    print("Watching for changes...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    init_logging()
    run_suggest_loop()
        
