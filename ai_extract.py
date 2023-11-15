import argparse
import os
import rope.base.project
from rope.base import libutils
from rope.refactor import extract
import instructor
from retry import retry
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, APITimeoutError

# Enables `response_model`
OPENAI_CLIENT = None
MODEL = "gpt-4-1106-preview" # 4 Turbo

def init_openai():
    global OPENAI_CLIENT
    OPENAI_CLIENT = instructor.patch(OpenAI(timeout=30.0))

def convert_line_range_to_offset(code, start, end):
    lines = rope.base.codeanalyze.SourceLinesAdapter(code)
    return lines.get_line_start(start), lines.get_line_end(end)

def parse_line_range(line_range: str | None) -> (int | None, int | None):
    if not line_range:
        return (None, None)
    parts = line_range.split('-')
    if len(parts) == 1:
         return (int(parts[0]), None)
    if len(parts) > 1:
        return int(parts[0]), int(parts[1])
    return (None, None)

def render_lines_with_numbers(lines: list[str]):
    return [str(i+1).rjust(5) + line for (i, line) in enumerate(lines)]

class ExtractSuggestion(BaseModel):
    start_line: int
    end_line: int
    name: str
    # critique: str = Field(description="Potential weaknesses of the suggestion")
    rating: int = Field(description="""
        Assessment of the improvement in readability created by the suggestion.
        Some ways to improve are labeling a concept that wasn't labeled before or splitting a large method into small ones.
        1 = little to none, 2 = some, 3 = large improvement
    """)

class ExtractSuggestions(BaseModel):
    suggestions: list[ExtractSuggestion]

@retry((ValidationError, APITimeoutError), tries=3, delay=1)
def suggest_extract(
    code
) -> list[ExtractSuggestion]:
    prompt = f"""
In the provided Python code, identify 4 extract method opportunities for a refactoring tool.
Suggest names that clearly describes the contents and avoid collision with other methods already present.
    
```
{code}
```
    """
    response = OPENAI_CLIENT.chat.completions.create(
        model=MODEL,
        response_model=ExtractSuggestions,
        temperature = 0.2,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.suggestions

def select_suggestion(lines, suggestions: list[ExtractSuggestion]) -> ExtractSuggestion | None:
    suggestions = sorted(suggestions, key=lambda suggestion: -suggestion.rating)
    print("Suggestions:")
    for suggestion in suggestions:
        print(suggestion)
        print()
        for line in lines[max(suggestion.start_line-3, 0) : suggestion.start_line+1]:
            print(line, end="")
        print(f"# Extract '{suggestion.name}' start")
        for line in lines[suggestion.start_line+1 : suggestion.end_line]:
            print(line, end="")
        print(f"# Extract '{suggestion.name}' end")
        for line in lines[suggestion.end_line : suggestion.end_line+3]:
            print(line, end="")
        print("Accept? y/n")
        choice = input().lower().strip()
        if choice in 'yY':
            return suggestion
    return None

def run_extract(file_name: str, line_range: str, name: str, project: str, ai: bool):
    start_line, end_line = parse_line_range(line_range)
    project = rope.base.project.Project(project)
    resource = libutils.path_to_resource(project, file_name)
    code = resource.read()
    code_lines_with_numbers = render_lines_with_numbers(code.splitlines(True))
    if ai:
        print("AI suggesting...")
        suggestions = suggest_extract(''.join(code_lines_with_numbers))
        suggestion = select_suggestion(code_lines_with_numbers, suggestions)
        if not suggestion:
            print("No suggestion")
            return False
        start_line = suggestion.start_line
        end_line = suggestion.end_line
        name = suggestion.name
    if not start_line:
        print("Missing start line")
        return False
    if not end_line:
        print("Missing end line")
        return False
    if not name:
        print("Missing name")
        return False
    print(f"Extracting {start_line}-{end_line} as {name}")
    start_offset, end_offset = convert_line_range_to_offset(code, start_line, end_line)
    extractor = extract.ExtractMethod(project, resource, start_offset, end_offset)
    changes = extractor.get_changes(name)
    project.do(changes)
    project.close()
    print(f"Consider: git commit -am 'Extract method {name}'")
    return True


def cli():
    parser = argparse.ArgumentParser(description='Extract method in python.')
    parser.add_argument('file', help='File to refactor')
    parser.add_argument('--project', help='Project directory (defaults to file dir)')
    parser.add_argument('-r', '--range', required=False, help='Line range, like 1-10. One indexed, inclusive.)')
    parser.add_argument('-n', '--name', required=False, help='Method name')
    parser.add_argument('--ai', required=False, help='Use GPT for name and range', action='store_true')

    args = parser.parse_args()
    project = args.project or os.path.dirname(os.path.abspath(args.file))
    success = run_extract(
        file_name=args.file, 
        line_range=args.range, 
        name=args.name, 
        project=project,
        ai=args.ai
    )
    if not success:
        exit(1)

if __name__ == '__main__':
    init_openai()
    cli()