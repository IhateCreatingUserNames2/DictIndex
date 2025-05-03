import os
import argparse


def create_directory_structure():
    """Create the directory structure for the project"""
    print("Creating directory structure...")

    # Create static directory if it doesn't exist
    if not os.path.exists("static"):
        os.makedirs("static")
        print("Created 'static' directory")

    # If codebase.txt doesn't exist, create an empty one
    if not os.path.exists("codebase.txt"):
        with open("codebase.txt", "w", encoding="utf-8") as f:
            f.write("# Índice de Ditador - Database File\n\n")
        print("Created empty 'codebase.txt' file")

    print("Directory structure created successfully!")


def setup_env_file(api_key=None):
    """Create .env file with API key"""
    if os.path.exists(".env"):
        print(".env file already exists, skipping...")
        return

    if api_key:
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("Created .env file with provided API key")
    else:
        with open(".env", "w", encoding="utf-8") as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        print("Created .env file template - Please edit and add your API key")


def main():
    parser = argparse.ArgumentParser(description="Setup the Índice de Ditador project")
    parser.add_argument("--api-key", help="Your OpenAI API key")

    args = parser.parse_args()

    print("====== Índice de Ditador - Setup ======")
    create_directory_structure()
    setup_env_file(args.api_key)

    print("\nSetup completed successfully!")
    print("\nTo start the application, run:")
    print("  uvicorn app:app --reload")
    print("\nThen visit http://localhost:8000 in your browser")
    print("=========================================")


if __name__ == "__main__":
    main()