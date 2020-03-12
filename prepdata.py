def prep(st,filter_type,freqmin,freqmax):
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type='cosine')
    st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
    return st
