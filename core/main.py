import os
import time
import logging

import chess
import chess.pgn

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
# from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('app.log') 
    ]
)
logger = logging.getLogger(__name__)

PROMPT = """
Given a PGN of the chess board, choose a legal move that corresponds to your strategy and respond in JSON format with a single field "move".
Do NOT use a move from the invalid_moves list.

Board: {board}

Legal moves: {legal_moves}

Invalid moves: {invalid_moves}

Your move:
"""

class Move(BaseModel):
    move: str = Field(description = "a valid chess move")


def get_pgn(board) -> str:
    pgn = str(chess.pgn.Game().from_board(board))
    return pgn

def try_move(player, board, pgn, legal_moves, invalid_moves = None):
    if invalid_moves is None:
        invalid_moves = []
        
    logger.debug(f"Board: {pgn}")
    logger.debug(f"Legal moves: {legal_moves}")
    logger.debug(f"Invalid moves: {invalid_moves}")

    move = player.invoke({"board": pgn, "legal_moves": legal_moves, "invalid_moves": str(invalid_moves)})
    move = move["move"]
    try:
        board.push_san(move)
        return move
    except:
        print(f"Invalid move: {move} \nTrying again")
        invalid_moves.append(move)
        time.sleep(1)
        try_move(player, board, pgn, legal_moves, invalid_moves)


def game_loop(board: chess.Board, llm1, llm2):
    prompt1 = ChatPromptTemplate.from_template(PROMPT)
    prompt2 = ChatPromptTemplate.from_template(PROMPT)
    # functions = [convert_to_openai_function(Move)]
    p1 = prompt1 | llm1 | JsonOutputParser(pydantic_object=Move)
    p2 = prompt2 | llm2 | JsonOutputParser(pydantic_object=Move)

    count = 0
    while True:
        pgn = get_pgn(board)
        legal_moves = str(list(board.legal_moves))
        try_move(p1, board, pgn, legal_moves)

        logger.info(f"Turn {count}\n")
        logger.info(f"\n{board}")

        pgn = get_pgn(board)
        legal_moves = str(list(board.legal_moves))
        try_move(p2, board, pgn, legal_moves)

        logger.info(f"\n{board}")
        time.sleep(3)

        count += 1
        if count == 1:
            new_strat_1 = input("Change strategy (p1): ")
            new_strat_prompt = f"""
            You are a chess player with the following strategy: {new_strat_1}\n
            """
            new_prompt = new_strat_prompt + PROMPT
            new_prompt_template = ChatPromptTemplate.from_template(new_prompt)
            p1 = new_prompt_template | llm1 | JsonOutputParser(pydantic_object=Move)


def main():
    # llm1 = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    llm1 = ChatOpenAI(model="gpt-4o", api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    # llm2 = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    llm2 = ChatOpenAI(model="gpt-4o", api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    game = chess.Board()

    game_loop(game, llm1, llm2)


if __name__ == "__main__":
    main()
