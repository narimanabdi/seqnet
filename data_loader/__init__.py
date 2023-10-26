from data_loader import belga2flick,belga2toplogo,gtsrb2tt100k,belga2toplogoft
from data_loader import gtsrb,gtsrb2toplogo,gtsrb2flick, tt100kft,belga2flickft,mini
def get_loader(name):
    return {
        'gtsrb': gtsrb,
        'gtsrb2tt100k': gtsrb2tt100k,
        'belga2toplogo': belga2toplogo,
        'belga2flick': belga2flick,
        'gtsrb2toplogo':gtsrb2toplogo,
        'gtsrb2flick':gtsrb2flick,
        'tt100kft':tt100kft,
        'belga2flickft':belga2flickft,
        'belga2toplogoft':belga2toplogoft,
        'mini':mini,
    }[name]