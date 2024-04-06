import asyncio
from typing import List

from dotenv import load_dotenv
import openai
import langmem
from pydantic import BaseModel, Field


load_dotenv()


class State(BaseModel):
    puzzle: str = Field(
        default=None,
        description="A state of the A::B Challenge puzzle in its string representation."
    )
    reason: str = Field(
        default=None,
        description="How we get into this state."
    )


class CurrentState(BaseModel):
    visited: List[State]= Field(
        default_factory=State,
        description="A list of all visited states when solving A::B Challenge. Each state include the puzzle string and the reason how we get into this state.",
    )
    current: State = Field(
        default_factory=State,
        description="The current state of the A::B Challenge puzzle."
    )
    experiences: str = Field(
        default=None,
        description="A condensed knowledge descriptions of rules, tips, and tricks learned from solving A::B Challenge."
    )


def load_rules():
    with open('rules.md', 'r') as file:
        rules = file.read()
    return rules


def solver_agent():
    return openai.AsyncClient()


def checker_agent():
    return openai.AsyncClient()


async def setup_memory():

    langmem_client = langmem.AsyncClient()

    # 1. user_state memory: set to current state
    current_state = await langmem_client.create_memory_function(
        CurrentState, target_type="user_state"
    )

    # 2. user_append_state memory: 
    replay_history = await langmem_client.create_memory_function(
        State, target_type="user_append_state"
    )
    # 3. user_semantic_memory (kg triplets): disable
    functions = langmem_client.list_memory_functions(target_type="user")

    semantic_memory = None
    async for func in functions:
        if func["type"] == "user_semantic_memory":
            semantic_memory = func

    if semantic_memory:
        await langmem_client.update_memory_function(
            semantic_memory["id"],
            status="disabled"
        )

    return langmem_client, current_state, replay_history


async def main(puzzle: str) -> str:
    rules = load_rules()
    solver = solver_agent()
    memory, current_state, replay_history = await setup_memory()

    messages = [
        {
            "role": "system",
            "content": f"You are a perfect computer trained to solve A::B Challenge.\n\n## Rules:\n{rules}\n\n## Instructions\n1. Please think step by step when you solve the puzzle.\n2. You will try to solve the puzzle\n3. You will scan each puzzle from left to right to determine if there's an opportunity to apply the rules.\n4. Iterate step 2 until no replacement rules can be applied.\n5. Take extreme care and always consider all available rules before.\n\nPlease be very careful, as 10000 dollars is on the line. If any mistake is made my grandma's live will be in danger."
        },
        {
            "role": "user",
            "content": "Let's start solving the following puzzle.\n\n{puzzle}"
        }
    ]

    response = await solver.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content


def get_user_input(prompt):
    return input(prompt)


if __name__ == "__main__":
    puzzle = get_user_input("Enter A::B Puzzle:\n>> ")
    print(f"A::B Puzzle:\n{puzzle}")
    steps = asyncio.run(main(puzzle))
    print(f"Solution:\n{steps}")
