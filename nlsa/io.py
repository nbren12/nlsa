import imp
def get_data(config, tag):
    kw = config['data'][tag]

    if "netcdf" in kw:
        import xarray
        v = xarray.open_dataset(kw['netcdf'])[kw['var']]
    elif "module" in kw:
        data = imp.load_source("data", kw['module'])
        v = getattr(data, kw['var'])

    t = kw['tdim']

    return v, t
