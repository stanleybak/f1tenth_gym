"""
Eval Speed of each Racer on an Empty test track
"""

import os
import tarfile

def main():
    """main entry point"""

    suffix = "-snapshot.tar.gz"

    for filename in os.listdir():
        if filename.endswith(suffix):
            if "Stony Brook" not in filename:
                continue # for debugging
            
            folder = filename[:-len(suffix)]

            print(folder)

            if not os.path.exists(folder):
                with tarfile.open(filename) as f:
                    f.extractall(os.path.join(".", folder))

            eval_driver_folder(folder)

def eval_driver_folder(folder):
    """evaluate the driver that was unzipped to the passed-in folder"""

    

if __name__ == "__main__":
    main()
