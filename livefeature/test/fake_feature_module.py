import livefeature as lf

@lf.feature("fake1", str)
def get_fake_feature(x):
    return "foobar"
