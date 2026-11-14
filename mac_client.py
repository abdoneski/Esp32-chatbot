#!/usr/bin/env python3
"""Send typed questions to the ESP32's local FAQ endpoint."""

import argparse
import urllib.error
import urllib.request

import llm_fallback

DEFAULT_ESP32_URL = "http://172.20.10.3"
COMMANDS = "/help, /health, /ask <question>, /llm [provider], /key <key>, /exit"


def ask_esp32(base_url: str, question: str, timeout: float) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ask",
        data=question.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8").strip()


def check_health(base_url: str, timeout: float) -> str:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/health")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8").strip()


def send_question(base_url: str, question: str, timeout: float) -> None:
    try:
        reply = ask_esp32(base_url, question, timeout)
        print(f"ESP32> {reply}")
        if (
            reply.startswith("I don't know that one yet")
            and llm_fallback.is_llm_available()
        ):
            print("ESP32> Asking LLM fallback...")
            try:
                llm_reply = llm_fallback.query_text(question, timeout=timeout)
                if llm_reply:
                    print(f"LLM> {llm_reply}")
            except Exception as error:
                print(f"LLM fallback failed: {error}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        print(f"ESP32 returned HTTP {error.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Could not reach the ESP32: {error}")
    except UnicodeDecodeError as error:
        print(f"ESP32 response was not valid UTF-8: {error}")


def handle_input(line: str, base_url: str, timeout: float) -> bool:
    """Handle a prompt line; return False only when the user requests exit."""
    message = line.strip()
    if not message:
        return True

    if not message.startswith("/"):
        send_question(base_url, message, timeout)
        return True

    command, _, argument = message.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        print(f"Commands: {COMMANDS}")
    elif command == "/health":
        try:
            print(f"ESP32> {check_health(base_url, timeout)}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            print(f"ESP32 returned HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"Could not reach the ESP32: {error}")
        except UnicodeDecodeError as error:
            print(f"ESP32 response was not valid UTF-8: {error}")
    elif command == "/ask":
        if argument:
            send_question(base_url, argument, timeout)
        else:
            print("Usage: /ask <question>")
    elif command == "/llm":
        if argument:
            llm_fallback.set_active_provider(argument)
        print(f"LLM Provider: {llm_fallback.get_status_description()}")
    elif command == "/key":
        if not argument:
            print("Usage: /key <api_key>")
            return True
        provider = llm_fallback.get_active_provider() or "gemini"
        llm_fallback.set_provider_key(provider, argument)
        print(f"Set API key for {provider}. Status: {llm_fallback.get_status_description()}")
    else:
        print(f"Unknown command: {command}. Commands: {COMMANDS}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the offline ESP32 FAQ assistant over your local Wi-Fi."
    )
    parser.add_argument(
        "esp32_url",
        nargs="?",
        default=DEFAULT_ESP32_URL,
        help=f"ESP32 base URL (default: {DEFAULT_ESP32_URL})",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    print(f"Connected target: {args.esp32_url}")
    print(f"Commands: {COMMANDS}")
    print(f"LLM fallback: {llm_fallback.get_status_description()}")

    while True:
        try:
            line = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not handle_input(line, args.esp32_url, args.timeout):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
