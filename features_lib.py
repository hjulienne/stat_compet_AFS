#library of functions to create new features

def spec_amp(spec):
    spec = spec.sum(1) # oh this is silly! we lose so much info!!
    spec_amp = (spec - spec.mean())/ spec.std() # we just keep the amplitude of spectrums
    return spec_amp
