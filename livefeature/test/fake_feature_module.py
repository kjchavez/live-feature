import livefeature as lf

@lf.feature("foobar", key="VoterParty", dtype=str)
def get_fake_feature(x):
    return "foobar"

@lf.feature("baz", key="VoterState", dtype=str)
def get_other_feature(x):
    return "baz"
