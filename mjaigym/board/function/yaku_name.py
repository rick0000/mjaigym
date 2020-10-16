from enum import Enum

class YakuName(Enum):
    Tenho = "tenho"
    Chiho = "chiho"
    Kokushimuso = "kokushimuso"
    Daisangen = "daisangen"
    Suanko = "suanko"
    Tsuiso = "tsuiso"
    Ryuiso = "ryuiso"
    Chinroto = "chinroto"
    Daisushi = "daisushi"
    Shosushi = "shosushi"
    Sukantsu = "sukantsu"
    Churenpoton = "churenpoton"
    
    Dora = "dora"
    Uradora = "uradora"
    Akadora = "akadora"

    Reach = "reach"
    Ippatsu = "ippatsu"
    MenzenchinTsumoho = "menzenchin_tsumoho"
    Tanyaochu = "tanyaochu"
    Pinfu = "pinfu"
    Ipeko = "ipeko"
    Sangenpai = "sangenpai"
    Bakaze = "bakaze"
    Jikaze = "jikaze"
    Rinshankaiho = "rinshankaiho"
    Chankan = "chankan"
    Haiteiraoyue = "haiteiraoyue"
    Hoteiraoyui = "hoteiraoyui"
    Sanshokudojun = "sanshokudojun"
    Ikkitsukan = "ikkitsukan"
    Honchantaiyao = "honchantaiyao"
    Chitoitsu = "chitoitsu"
    Toitoiho = "toitoiho"
    Sananko = "sananko"
    Honroto = "honroto"
    Sanshokudoko = "sanshokudoko"
    Sankantsu = "sankantsu"
    Shosangen = "shosangen"
    DoubleReach = "double_reach"
    Honiso = "honiso"
    Junchantaiyao = "junchantaiyao"
    Ryanpeko = "ryanpeko"
    Chiniso = "chiniso"

YAKU_CHANNEL_CONVERT_LIST = [
    YakuName.Tenho,
    YakuName.Chiho,
    YakuName.Kokushimuso,
    YakuName.Daisangen,
    YakuName.Suanko,
    YakuName.Tsuiso,
    YakuName.Ryuiso,
    YakuName.Chinroto,
    YakuName.Daisushi,
    YakuName.Shosushi,
    YakuName.Sukantsu,
    YakuName.Churenpoton,
    YakuName.Reach,
    YakuName.Ippatsu,
    YakuName.MenzenchinTsumoho,
    YakuName.Tanyaochu,
    YakuName.Pinfu,
    YakuName.Ipeko,
    YakuName.Sangenpai,
    YakuName.Bakaze,
    YakuName.Jikaze,
    YakuName.Rinshankaiho,
    YakuName.Chankan,
    YakuName.Haiteiraoyue,
    YakuName.Hoteiraoyui,
    YakuName.Sanshokudojun,
    YakuName.Ikkitsukan,
    YakuName.Honchantaiyao,
    YakuName.Chitoitsu,
    YakuName.Toitoiho,
    YakuName.Sananko,
    YakuName.Honroto,
    YakuName.Sanshokudoko,
    YakuName.Sankantsu,
    YakuName.Shosangen,
    YakuName.DoubleReach,
    YakuName.Honiso,
    YakuName.Junchantaiyao,
    YakuName.Ryanpeko,
    YakuName.Chiniso,
]
YAKU_CHANNEL_CONVERT_LIST = [v.value for v in YAKU_CHANNEL_CONVERT_LIST]
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "1")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "2")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "3")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "4")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "5")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "6")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "7")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "8")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "9")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "10")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "11")
YAKU_CHANNEL_CONVERT_LIST.append(YakuName.Dora.value + "12")

YAKU_CHANNEL_MAP = dict(zip(YAKU_CHANNEL_CONVERT_LIST, range(len(YAKU_CHANNEL_CONVERT_LIST))))

