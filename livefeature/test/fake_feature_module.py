import livefeature as lf

@lf.feature("fake1", key="VoterParty", dtype=str)
def get_fake_feature(x):
    return "foobar"
