import os
import os.path
import sys
from subprocess import call

def main():
    path = 'C:\\Users\\kersc\\Desktop\\dialectdetect\\audio'
    filenames = [
        filename
        for filename
        in os.listdir(path)
        if filename.endswith('.mp3')
        ]
    for filename in filenames:
        call(['lame', '-V0',
              os.path.join(path, filename),
              os.path.join(path, '%s.wav' % filename[:-4])
              ])
    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)