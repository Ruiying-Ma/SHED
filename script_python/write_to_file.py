import os
import json

def write_to_file(dest_path: str, contents, is_append, is_json):
    '''
    `dest_path`: absolute path
    '''
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if is_append:
        assert is_json == False
        with open(dest_path, 'a') as file:
            file.write(contents)
    else:
        if is_json:
            assert dest_path.endswith(".json")
            with open(dest_path, 'w') as file:
                json.dump(contents, file, indent=4)
        else:
            with open(dest_path, 'w') as file:
                file.write(contents)