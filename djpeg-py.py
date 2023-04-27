
import os
import subprocess
import sys

def main():

    # cmd = 'djpeg DSC_3459.JPG > test.ppm'
    print(cmd)
    subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)


if "__main__" == __name__:
    main()
