import os
import zipfile


def zip_folder(folder_path: str, output_path: str) -> bool:
    """
    Zip a folder and its contents to a zip file.

    Args:
        folder_path (str): Path to the folder to be zipped
        output_path (str): Path where the zip file will be created

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the folder exists
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a valid directory")
            return False

        # Get the absolute path of the folder
        abs_folder_path = os.path.abspath(folder_path)

        # Create a ZipFile object in write mode
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the folder
            for root, dirs, files in os.walk(abs_folder_path):
                for file in files:
                    # Get the absolute path of the file
                    abs_file_path = os.path.join(root, file)

                    # Calculate relative path for the file inside the zip
                    rel_path = os.path.relpath(
                        abs_file_path, os.path.dirname(abs_folder_path)
                    )

                    # Add file to zip
                    zipf.write(abs_file_path, rel_path)

        print(f"Successfully created zip file at {output_path}")
        return True

    except Exception as e:
        print(f"Error creating zip file: {e}")
        return False
