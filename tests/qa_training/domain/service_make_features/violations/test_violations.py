import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert


@pytest.fixture()
def fixture_violations_pclass():
    service = ServiceMakeFeatures()
    df_in = pd.read_csv("tests/qa_training/domain/service_make_features/violations/df_violations_pclass.csv")
    df_out_expected = pd.read_csv("tests/qa_training/domain/service_make_features/violations/df_violations_pclass_expected.csv")
    return service, df_in, df_out_expected

def fixture_violations_():
    service = ServiceMakeFeatures()
    df_in = pd.read_csv("tests/qa_training/domain/service_make_features/violations/df_violations_pclass.csv")
    df_out_expected = pd.read_csv("tests/qa_training/domain/service_make_features/violations/df_violations_pclass_expected.csv")
    return service, df_in, df_out_expected

def test_violations_pclass(fixture_violations_pclass):
    service, df_in, df_out_expected = fixture_violations_pclass
    df_out = service._handle_violations(df_in)
    MyAssert().assert_df(df_out, df_out_expected)
