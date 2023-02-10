from data_loader import belga2flick,belga2toplogo,gtsrb2tt100k
from data_loader import gtsrb,mini,gtsrb2flick
def get_loader(name):
    return {
        'gtsrb': gtsrb,
        'gtsrb2tt100k': gtsrb2tt100k,
        'belga2toplogo': belga2toplogo,
        'belga2flick': belga2flick,
        'mini':mini,
        'gtsrb2flick':gtsrb2flick
    }[name]