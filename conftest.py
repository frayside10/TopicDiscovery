import pytest
import pandas as pd

@pytest.fixture
def stop_word_list():
    return ['fred','bob','hazel']


@pytest.fixture
def raw_words_for_tokenizing():
    values = {'text': ['PROBABLY THE WORST ***!??? film!!##!!???! Ever! ']}
    df_for_check = pd.DataFrame(values, columns=['text'])
    return df_for_check


@pytest.fixture
def raw_words_for_bigram():
    values = [['sci', 'fi', 'sci','fi','sci','fi','sci', 'fi', 'sci','fi','sci','fi','sci', 'fi', 'sci','fi','sci','fi','sci', 'fi', 'sci','fi','sci','fi','sci', 'fi', 'sci','fi','sci','fi','sci', 'fi', 'sci','fi','sci','fi'], ['sci', 'fi', 'sci','fi','sci','fi']]
    return values

