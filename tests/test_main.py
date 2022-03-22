import os


from business_logic.main import compute_corrispondence_from_image_google  # affinche funzioni l'import va fatto Mark as source della cartella business_logic


# The __init__.py file indicates that the input_files in a folder are part of a Python package.
# Without an __init__.py file, you cannot import input_files from another directory in a Python project.

def test_compute_corrispondence_from_image_google():

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    compute_corrispondence_from_image_google(ROOT_DIR + '/pagine')
