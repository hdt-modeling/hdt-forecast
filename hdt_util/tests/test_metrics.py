from hdt_util.metrics import W1


def test_W1():
    #comparing distributions without a delay
    p1 = [1, 2, 3, 4]
    p2 = [1, 2, 3, 4]
    delay_dist = [1]
    d = W1(p1, p2, delay_dist)
    assert d == 0, "p1 and p2 are equal, but W1 equals {}".format(d)

    p1 = [1, 0]
    p2 = [0, 1]
    delay_dist = [1]
    d = W1(p1, p2, delay_dist)
    assert d == 1, "W1 should be {} but got {}".format(1, d)

    p1 = [1, 0, 0]
    p2 = [0, 0, 1]
    delay_dist = [1]
    d = W1(p1, p2, delay_dist)
    assert d == 2, "W1 should be {} but got {}".format(2, d)

    #comparing distributions with a delay
    #Note that only p2 is being convolved with the delay distribution
    p1 = [1, 0, 0]
    p2 = [0, 1, 1]
    delay_dist = [1/2,1/2]
    d = W1(p1, p2, delay_dist)
    assert d == 5/3, "W1 should be {} but got {}".format(5/3, d)

    p1 = [1, 1, 0]
    p2 = [0, 1, 1]
    delay_dist = [1/2,1/2]
    d = W1(p1, p2, delay_dist)
    assert d == 7/6, "W1 should be {} but got {}".format(7/6, d)

    p1 = [1, 1/2, 0]
    p2 = [0, 1, 1]
    delay_dist = [1/2,1/2]
    d = W1(p1, p2, delay_dist)
    d = round(d, 8)
    assert d == round(4/3, 8), "W1 should be {} but got {}".format(round(4/3,8), d)

    p1 = [1, 0, 0, 0]
    p2 = [0, 0, 1, 1]
    delay_dist = [1/2,1/4,1/4]
    d = W1(p1, p2, delay_dist)
    assert d == 13/5, "W1 should be {} but got {}".format(13/5, d)
