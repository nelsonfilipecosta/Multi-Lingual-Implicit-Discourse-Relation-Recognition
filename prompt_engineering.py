import sys
import pandas as pd


MODE = sys.argv[1]
if MODE not in ['validation', 'test']:
    print('Type a valid mode: validation or test.')
    exit()

LANG = sys.argv[2]
if LANG not in ['all', 'en', 'de', 'fr', 'cs']:
    print('Type a valid language: all, en, de, fr or cs.')
    exit()


ENG_CONNECTORS = ['at the same time', 'then', 'after', 'because', 'as a result', 'in that case', 'if', 'if not', 'unless',
                  'for that purpose', 'so that', 'even though', 'nonetheless', 'on the other hand', 'similarly', 'also',
                  'or', 'in other words', 'other than that', 'an exception is that', 'this illustrates that', 'for example',
                  'in short', 'in more detail', 'thereby', 'as if', 'rather than', 'instead']

GER_CONNECTORS = ['gleichzeitig', 'dann', 'davor,', 'weil', 'daher', 'insofern', 'wenn', 'sonst', 'es sei denn', 'dazu',
                  'sodass', 'obwohl', 'trotzdem', 'andererseits', 'gleichermaßen', 'darüberhinaus', 'oder', 'anders gesagt',
                  'abgesehen von dieser Ausnahme', 'eine Ausnahme ist, dass', 'das verdeutlicht, dass', 'zum Beispiel',
                  'um es kurz zu machen', 'genauer gesagt', 'hiermit', 'mittels', 'anstatt, dass', 'stattdessen']

FRE_CONNECTORS = ['en même temps', 'ensuite', 'après que', 'parce que', "c'est pourquoi", "dans ce cas", "si", "sinon",
                  "à moins que", "à cette fin", "afin que", "bien que", "néanmoins", "d'autre part", "de même", "en plus",
                  "ou", "en d'autres termes", "à part ça", "une exception est que", "cela illustre que", "par exemple",
                  "bref", "plus précisement", "de cette manière", "comme si", "plutôt que", "au lieu de"]

CZE_CONNECTORS = ["zároveň", "potom", "předtím", "protože", "prozo", "v tom případě", "pokud", "jinak", "ledaže",
                  "za tím účelem", "aby", "a to i přesto, že", "přesto", "na druhou stranu", "podobně", "také", "nebo",
                  "jinými slovy", "kromě této výjimky", "výjimkou je to, že", "to je příkladem toho, že", "například",
                  "zkrátka", "konkrétně", "tímto způsobem", "následujícím způsobem", "místo, aby", "místo toho"]


if MODE == 'validation':
    df = pd.read_csv('Data/DiscoGeM-2.0/discogem_2_single_lang_' + LANG + '_validation.csv')
elif MODE == 'test':
    df = pd.read_csv('Data/DiscoGeM-2.0/discogem_2_single_lang_' + LANG + '_test.csv')

arg_1 = df['arg1'].tolist()
arg_2 = df['arg2'].tolist()
labels = df.iloc[:,32:60].values.tolist()