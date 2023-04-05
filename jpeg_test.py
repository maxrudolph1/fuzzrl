import simplejpeg as sj
import os
import subprocess
import coverage
import numpy as np
import pylint_runner
# with open('test.jpeg', 'rb') as f:
#     img = sj.decode_jpeg(f.read())



result = pylint_runner.run(['--enable=branch-coverage', 'djpeg-py.py', 'main'])



'''

# bashcomand = "djpeg test.jpeg"
cmd = 'coverage run --branch djpeg-py.py DSC_3459.JPG'
subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
cmd = 'coverage report -m'
results = subprocess.run(cmd.split(" "), capture_output=True, text=True)

print(results.stdout)
# cov.stop()
# cov.save()

# cov.html_report()
'''