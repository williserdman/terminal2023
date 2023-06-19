import os
import json
import pickle

def parse_orig_replay_file():
    # Define the directory path where the replay files are located
    directory = "replays"

    # Get the list of files in the directory
    file_list = [filename for filename in os.listdir(directory) if filename.endswith(".replay")]

    # Loop through each file
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Initialize variables
        extracted_lines = []
        include_next_line = False

        # Loop through each line
        for line in lines:
            # Extract the JSON object
            try:
                json_obj = json.loads(line)
            except:
                print("line skipped")
                continue

            # Check the "turnInfo" array in the 0th index
            if "turnInfo" in json_obj and json_obj["turnInfo"][0] == 0:
                # Include the current line and set the flag to include the next line
                extracted_lines.append(line.strip())
                include_next_line = True
            elif include_next_line:
                # Include the next line
                extracted_lines.append(line.strip())
                include_next_line = False

        # Write the extracted lines to a new file
        output_file_path = os.path.splitext(file_path)[0] + "_extracted.replay"
        with open(output_file_path, "w") as output_file:
            output_file.write("\n".join(extracted_lines))

        print(f"Processed {filename} and saved extracted lines to {output_file_path}")
        return output_file_path

def get_moves_from_input(f_path):
    with open(f_path, "r") as f:
        lines = f.readlines()
    
    keyss = []
    valss = []
    i = 0
    while i < len(lines):
        jl = json.loads(lines[i])
        orig_units = jl["p1Units"]
        jl2 = json.loads(lines[i + 1])
        new_units = jl2["p1Units"]
        confirmed_new_units = []
        
        # the unit lists will always be 8 items long
        for j in range(len(orig_units)):
            a = orig_units[j]
            b = new_units[j]
            if len(a) == len(b):
                continue
            for k in range(len(b[len(a):])):
                confirmed_new_units += [j, b[k][0], b[k][1]]
        
        if len(confirmed_new_units) <= 60*3:
            confirmed_new_units += [0, 0, 0]*int(60-len(confirmed_new_units)/3)
        else:
            confirmed_new_units = confirmed_new_units[:60*3]

        keyss.append(lines[i])
        valss.append(confirmed_new_units)
        print(len(confirmed_new_units))
        i += 2
    
    with open(f_path + "_pairs.pickle", "wb") as f:
        pickle.dump([keyss, valss], f)



if __name__ == "__main__":
    get_moves_from_input(parse_orig_replay_file())

