import os
import sys
import argparse

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from colored import Fore, Style

from trs.main import TRS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='trs-cli',
        description='Chat with and summarize CTI reports (Local LLaMA3)'
    )

    parser.add_argument(
        '-c', '--chat',
        required=True,
        action='store_true',
        help='Enter chat mode'
    )

    args = parser.parse_args()

    # ❌ REMOVE OPENAI KEY
    # ✔️ ADD LOCAL MODEL NAME
    MODEL_NAME = "llama3"

    trs = TRS(model_name=MODEL_NAME, provider="ollama")

    COMMAND_HANDLERS = {
        '!summ': trs.summarize,
        '!detect': trs.detections,
        '!custom': trs.custom
    }

    if args.chat:
        console = Console()
        print(f'{Style.BOLD}{Fore.cyan_3}commands:{Style.reset}')
        print(f'* {Fore.cyan_3}!summ <url>{Style.reset} - summarize a threat report')
        print(f'* {Fore.cyan_3}!detect <url>{Style.reset} - identify detections in report')
        print(f'* {Fore.cyan_3}!custom <prompt_name> <url>{Style.reset} - custom prompt')
        print(f'* {Fore.cyan_3}!exit|!quit{Style.reset} - exit')

        print(f'{Style.BOLD}{Fore.dark_orange_3b}ready to chat with LLaMA-3!{Style.reset}\n')

        try:
            while True:
                prompt = input('💀 >> ').strip()

                if prompt.lower() in ['!exit', '!quit', '!q', '!x']:
                    logger.info('Exiting')
                    break

                command, *args = prompt.split()
                handler = COMMAND_HANDLERS.get(command.lower())

                if handler:
                    result = handler(*args)

                    if command.lower() == '!summ':
                        summary, mindmap, iocs = result
                        print('🤖 >>')
                        console.print(Markdown(summary))
                        console.print(Markdown(mindmap))
                        print(iocs)
                    else:
                        print('🤖 >>')
                        console.print(Markdown(result))
                else:
                    result = trs.qna(prompt=prompt)
                    print('🤖 >>')
                    console.print(Markdown(result))

        except KeyboardInterrupt:
            logger.info('Keyboard interrupt — exiting')
            sys.exit(0)
        except Exception as err:
            logger.error(f'error: {err}')
            sys.exit(1)
