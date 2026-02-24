from xgbwwdata import Filters

def test_filters_construct():
    f = Filters()
    assert f.min_rows == 200
