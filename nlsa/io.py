import imp
def get_data(config, tag):
    kw = config['data'][tag]
    data = imp.load_source("data", kw['module'])
    v = getattr(data, kw['var'])
    t = kw['tdim']

    return v, t
