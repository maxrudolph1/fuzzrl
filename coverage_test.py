import coverage

def testfunc(a):
    if a == 0:
        return 0
    else:
        for i in range(10):
            a += 1
        return a

cov = coverage.Coverage()
cov.start()

testfunc(1)

cov.stop()

out = cov.report()

print(out)


# cov.html_report('.coverage')