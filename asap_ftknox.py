from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.dates import date2num
import numpy as np
from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation, for some reason it doesn't accept fdsn
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing
from obspy.core.event import read_events
from obspy.core.event import ResourceIdentifier
# could also do:
# from obspy.core.event import *
from obspy import UTCDateTime
from obspy import read_inventory, read_events
from obspy import read
from matplotlib.ticker import FormatStrFormatter
# -*- coding: utf-8 -*-
from obspy.taup import TauPyModel
from obspy.taup import plot_travel_times

#CHOOSE FROM LOCAL V MODELS FOR TAUP CALCULATION
#model_t = TauPyModel(model="ak135")
model_l = TauPyModel(model="northak")
#model_l = TauPyModel(model="scak")
#model_l = TauPyModel(model="kaktovik")


# 2012 - 2019 M1.4-M2.0
event_list = ['ak019gb3spit', 'ak019fj0abqd', 'ak019fe1dspa', 'ak019faqodxb', 'ak019f2s778h', 'ak019esj5eq3', 'ak019ep7zkz6', 'ak019ec0dw7p', 'ak019e2g5nw2', 'ak019dk9hexo', 'ak019dilwmhg', 'ak019dbzjvpi', 'ak019d6nayrz', 'ak019d0ewb85', 'ak019cydo9b7', 'ak019bydhtnx', 'ak019bhu4skb', 'ak019bej3t33', 'ak019b4lt632', 'ak019ao2fi7b', 'ak019ahgbnkj', 'ak0199ub0129', 'ak0199fexdm2', 'ak019926mir6', 'ak0198jzqtt6', 'ak0198bq9lgg', 'ak0198a2i2ia', 'ak01988ezegx', 'ak0197ok3dlp', 'ak0197o6tbpp', 'ak0197jm07s5', 'ak0197hyakr2', 'ak0197endg4g', 'ak0197bb7r2n', 'ak01974cd1v3', 'ak01971f0566', 'ak0196ustpqt', 'ak0196o6q9h1', 'ak0196j869vv', 'ak0195khs6p2', 'ak019579hlkk', 'ak0194igdug6', 'ak0194dhxp1l', 'ak019409noi1', 'ak0193vb4fx2', 'ak0193tnk8sf', 'ak0193oorew4', 'ak0193n1fj9v', 'ak0193gf4ctv', 'ak0193645djw', 'ak0192ztwos7', 'ak0192uk8f4p', 'ak0192svu6pj', 'ak0192r8f9w8', 'ak0192fnpey1', 'ak0192ccjybk', 'ak0191k8b6p7', 'ak0191flmdyz', 'ak0191dmvya6', 'ak01915chy46', 'ak01913p41rw', 'ak019x3022i', 'ak019liahkh', 'ak019gjf8ns', 'ak019a961x8', 'ak0196m6ote', 'ak018grz6v7n', 'ak018gpzol0x', 'ak018goc6xid', 'ak018gl1p1zm', 'ak018g2u630e', 'ak018fplw803', 'ak018fnyi0l9', 'ak018fj0ekzu', 'ak018f93ba76', 'ak018f45ala5', 'ak018egyevk5', 'ak018dz4hrwr', 'ak018dsizckv', 'ak018cd7t33j', 'ak20255078', 'ak018brp1qqh', 'ak0189xlv6v8', 'ak0188yvizof', 'ak0188jzop6f', 'ak0188gb4an7', 'ak01883gc87a', 'ak0187tiq6tr', 'ak0187em7n4p', 'ak01879om3o9', 'ak01871ez43f', 'ak01841tf0xz', 'ak0183ldm623', 'ak0183bgf3y1', 'ak0182iylwt1', 'ak0182e06f9e', 'ak0181quhgis', 'ak0181ilhiqf', 'ak01815ch0zh', 'ak018vg0qua', 'ak017g9gg6tv', 'ak017g7sqnn6', 'ak017fw7yqk2', 'ak017fsx3u07', 'ak017fpzu749', 'ak017fny9zka', 'ak017fcdjxf0', 'ak017f92hme6', 'ak017f0ss0tw', 'ak017eqw9o4c', 'ak017ep81b43', 'ak017egybpy6', 'ak017e43w306', 'ak017ds5cfze', 'ak017dilr1bm', 'ak017dgyh39m', 'ak017dfan041', 'ak017c3c5jns', 'ak017btetwc7', 'ak017bhgmff4', 'ak017be5dxqr', 'ak017b69ajri', 'ak017at156s3', 'ak017a7j26to', 'ak017a5vgofz', 'ak0179m0po0b', 'ak01796sqjb5', 'ak0178twzhsb', 'ak01752au0t6', 'ak0174vmlti2', 'ak0174nezs1l', 'ak0173gf0o3u', 'ak0173erjnmf', 'ak0173d3ytxe', 'ak017385f6ft', 'ak0171m9uu79', 'ak017ot589w', 'ak017li0nuw', 'ak017gjejx0', 'ak017d8c2h6', 'ak016gn2lafq', 'ak016gb442lm', 'ak016g9gauex', 'ak016g6ixj8u', 'ak016g6569eq', 'ak016g17g4by', 'ak016fr9czq7', 'ak016fpmerq8', 'ak016fnydoge', 'ak016fe1c1ya', 'ak016eswcfd6', 'ak016ep80uon', 'ak016e7et0dx', 'ak016dvrmx24', 'ak016dnkb5uc', 'ak016dlwty7h', 'ak016dilugw9', 'ak016dgy8nu5', 'ak016dbzlchf', 'ak016d710grb', 'ak016cye6fjx', 'ak016cotiksm', 'ak016cblt3k2', 'ak016btew79l', 'ak016bjhl6ee', 'ak016b2y9yz5', 'ak016aznb6i8', 'ak016avyfm0z', 'ak016ardiuh0', 'ak016achngl6', 'ak0169z9gr1g', 'ak0169nole09', 'ak0169h2kpgf', 'ak016975aq5b', 'ak0168d03al3', 'ak0167tj3hw8', 'ak0167q7zrnv', 'ak0167okhou9', 'ak0167mx8uvp', 'ak0167gatkwu', 'ak0167en96q6', 'ak0167ayq34u', 'ak0166zrfq2e', 'ak0166miu2al', 'ak0166clx82b', 'ak01665zqux6', 'ak01662op608', 'ak0165xq16w9', 'ak0165r1hn1a', 'ak0165dv80d7', 'ak01658x0qqa', 'ak01653ye359', 'ak01645833g4', 'ak01643idzzu', 'ak01641x0m4u', 'ak0163tncztd', 'ak0163rzsiob', 'ak0163ldp28u', 'ak0163jqaauc', 'ak0163i2m6fr', 'ak0163gf0g33', 'ak0163g34tz5', 'ak0163ep527m', 'ak01639q6w3c', 'ak016365wc5m', 'ak0162ujg27p', 'ak01627duthi', 'ak0162145dnj', 'ak0161w4y5v7', 'ak0161quhcug', 'ak0161p71v4a', 'ak0161k8f180', 'ak0161gx8161', 'ak01618nogqo', 'ak01613oyhcm', 'ak016qgnyc1', 'ak016n5jyns', 'ak016juhgcu', 'ak0169x8u62', 'ak0163b3e7b', 'ak015grncdlk', 'ak015gcs3gh3', 'ak015g7srz6f', 'ak015g67aj6w', 'ak015g4if7d7', 'ak015g2uq1i2', 'ak015fxw8bhy', 'ak015fuwbhw1', 'ak015frnd9d1', 'ak015fizp9ta', 'ak015f62st0f', 'ak015f2sx6cj', 'ak015exuwagg', 'ak015eu6mocn', 'ak015eqxw7fl', 'ak015ep7zz11', 'ak015elwwh4h', 'ak015eh04dck', 'ak015edn8dwc', 'ak015eaq6loc', 'ak015dp7wkno', 'ak015dlwtj7d', 'ak015dfacwo6', 'ak015df71yyh', 'ak015dbzs2vp', 'ak015d8myx51', 'ak015d5dh8d4', 'ak015cou85ad', 'ak015cbk5m0d', 'ak015c7x8a88', 'ak015btexyz0', 'ak015bq3ruqk', 'ak015bog8i2c', 'ak015b9k4aqx', 'ak015b69b7f4', 'ak015b0yufa5', 'ak015axx8iih', 'ak015akrm704', 'ak015ae57627', 'ak015achmqw9', 'ak0159xm1wop', 'ak0159ncrxcy', 'ak015926kzum', 'ak01590j25ks', 'ak0158a0we34', 'ak01588ey5b4', 'ak0157czoscd', 'ak0157bc5o9d', 'ak0156avf5px', 'ak01564c67pq', 'ak0155xq9emo', 'ak0155sp0yqo', 'ak01553veu3h', 'ak0154lrgdy3', 'ak0154ig43hi', 'ak0154gsw7n2', 'ak01543klysk', 'ak0153jq2rd7', 'ak01534uc01e', 'ak0152r89xim', 'ak0152m9pe2s', 'ak0152km4iey', 'ak0152cch2gb', 'ak01522f90p1', 'ak0151vt2kxu', 'ak0151si04i5', 'ak0151pitt75', 'ak0151gx7s9w', 'ak0151bymlyz', 'ak0151700l96', 'ak01515choxm', 'ak015z2wwq8', 'ak015vf9t05', 'ak015qhhpyx', 'ak015gjecvt', 'ak0156m86kr', 'ak014goc5ccc', 'ak014ghpyxbb', 'ak014gg2f4o9', 'ak014g1klf7t', 'ak014fxvjest', 'ak014fe1o88l', 'ak014faq0bef', 'ak014f2ubbs0', 'ak014f2g9ye8', 'ak014eu6soup', 'ak014eqvipm8', 'ak014elwzdzf', 'ak014eaq6edc', 'ak014eac6buz', 'ak014dilp4n9', 'ak014dda04m6', 'ak014d5dfdsm', 'ak014cvg67cz', 'ak014cliz8xa', 'ak014cblrxnr', 'ak014axzmoml', 'ak0147utm51t', 'ak0146usu3yg', 'ak0146kvjxjh', 'ak0145m59n7c', 'ak014579ghu3', 'ak0144u14qpm', 'ak0144bu8j12', 'ak0143l0630c', 'ak01432t1jv2', 'ak0142zi6ktn', 'ak0142codxvp', 'ak01420rrd49', 'ak0141si0it9', 'ak0141ikv97e', 'ak0141flljop', 'ak014evw57b', 'ak014d8goe5', 'ak014bkvj2f', 'ak013ghpypm7', 'ak013fny9og3', 'ak013extpagx', 'ak013eiltxum', 'ak013dnkb3jo', 'ak013cn6jocn', 'ak013bl55t4b', 'ak013bb7xq7s', 'ak013a5t4cdk', 'ak013a0wvza3', 'ak0139uap96v', 'ak0139qznfhi', 'ak0138ddkm7s', 'ak0137gas2hs', 'ak01376dkogj', 'ak0136y3wbl1', 'ak0136kvln9g', 'ak0136j81mzc', 'ak0136fx704q', 'ak0136aycmbw', 'ak01358wzau3', 'ak01355lwata', 'ak0134p2jcck', 'ak0134bue9qr', 'ak01341un6j5', 'ak0133jq47x5', 'ak0132ztvdm6', 'ak0132svvx2q', 'ak0132r8dea9', 'ak0132fnl9bc', 'ak0132dzz4ri', 'ak013242t91p', 'ak0131siad04', 'ak0131qujot9', 'ak0131p711cc', 'ak0131ikrzj6', 'ak0131702wgl', 'ak013121eb6w', 'ak013vgmuh3', 'ak013ew0u0m', 'ak013d8dcsj', 'ak01389qhz2', 'ak0136m6c7a', 'ak0133b3zce', 'ak012gpzpsvu', 'ak012gb3u1jp', 'ak012g9g94s5', 'ak012g7sr0zx', 'ak012fr9du2q', 'ak012f0ssp3m', 'ak012eqw8101', 'ak012efbcpem', 'ak012edn84v5', 'ak012e3q61df', 'ak012dz54n1f', 'ak012dxhknkx', 'ak012ddn4a1t', 'ak012d0et9wn', 'ak012cou8l54', 'ak012c4zlo16', 'ak012c3c3a74', 'ak012c1b0qfz', 'ak012bwc8o89', 'ak012bteugoh', 'ak012bg6oezb', 'ak012ard6upi', 'ak012ahg96cb', 'ak0129qzmziv', 'ak0128vk67vr', 'ak0128twx4jk', 'ak01288cjihs', 'ak01286qvmq5', 'ak012853tfcj', 'ak0128058r6k', 'ak0127v6llwz', 'ak0127mwxj2s', 'ak0127bc6is9', 'ak01262m9jfn', 'ak0125w2hq4c', 'ak0125nt39pg', 'ak0125khrinw', 'ak01253ydgxv', 'ak01252aurit', 'ak0124lrl4z3', 'ak0123n164ht', 'ak01232t4smv', 'ak0122r89krx', 'ak0122fnlrsc', 'ak01225qatfc', 'ak0121vt3w5d', 'ak019fe1dspa', 'ak019esj5eq3', 'ak019ep7zkz6', 'ak019e2g5nw2', 'ak019da9v096', 'ak019ao2fi7b', 'ak019akrdner', 'ak0198qlzw0y', 'ak0197jm07s5', 'ak01971f0566', 'ak019579hlkk', 'ak019528l3mh', 'ak019409noi1', 'ak0193gf4ctv', 'ak0192ztwos7', 'ak0192svu6pj', 'ak0191xgm3i2', 'ak0191flmdyz', 'ak0191dmvya6', 'ak01913p41rw', 'ak0199xbirr', 'ak0196m6ote', 'ak018goc6xid', 'ak018dsizckv', 'ak20255078', 'ak0188gb4an7', 'ak01871ez43f', 'ak0182649r78', 'ak018vg0qua', 'ak017gb3zzk7', 'ak017fsx3u07', 'ak017f92hme6', 'ak017ew85y90', 'ak017dgyh39m', 'ak017c3c5jns', 'ak017be5dxqr', 'ak017at156s3', 'ak017a5vgofz', 'ak0179m0po0b', 'ak01752au0t6', 'ak0174vmlti2', 'ak0174nezs1l', 'ak0173erjnmf', 'ak0173d3ytxe', 'ak017385f6ft', 'ak01734gkp9s', 'ak017291cq01', 'ak0171m9uu79', 'ak01717c4iyc', 'ak01712dazn6', 'ak017vfdoea', 'ak017ot589w', 'ak017li0nuw', 'ak017juhgga', 'ak017gjejx0', 'ak017d8c2h6', 'ak016gpzra6c', 'ak016gn2lafq', 'ak016gb442lm', 'ak016g9gauex', 'ak016g6569eq', 'ak016g17g4by', 'ak016fsxivdz', 'ak016fr9czq7', 'ak016fnydoge', 'ak016fhc4oos', 'ak016dvu2vd3', 'ak016dlwty7h', 'ak016dbzlchf', 'ak016d710grb', 'ak016bjhl6ee', 'ak016avyfm0z', 'ak0168f1a5jr', 'ak0168d03al3', 'ak0167tj3hw8', 'ak0167q7zrnv', 'ak0167ayq34u', 'ak0166miu2al', 'ak0166j5dl9h', 'ak01665zqux6', 'ak01662op608', 'ak0165zdl9m4', 'ak0165xq16w9', 'ak0165dv80d7', 'ak01658x0qqa', 'ak0164yzrros', 'ak0164p2jau1', 'ak0164jqgnfc', 'ak0164f5142k', 'ak01643idzzu', 'ak01641x0m4u', 'ak0163vawk34', 'ak0163tncztd', 'ak0163jqaauc', 'ak0163i2m6fr', 'ak0163gf0g33', 'ak0163g34tz5', 'ak0163cqf8nj', 'ak01639q6w3c', 'ak0162ztupo9', 'ak0162ujg27p', 'ak0162pkxi6j', 'ak0162izd1rg', 'ak0162hb1psf', 'ak0162fnhghy', 'ak01625qjo3u', 'ak0162145dnj', 'ak0161z46crq', 'ak0161quhcug', 'ak0161p71v4a', 'ak0161gx8161', 'ak0161cb4c7m', 'ak01618nogqo', 'ak01613oyhcm', 'ak016s45vry', 'ak016qgnyc1', 'ak016juhgcu', 'ak016gjhe0c', 'ak016a9wwsl', 'ak0169x8u62', 'ak0163b3e7b', 'ak015g7srz6f', 'ak015g67aj6w', 'ak015g16kfoy', 'ak015fxw8bhy', 'ak015fuwbhw1', 'ak015fra581a', 'ak015fnybbut', 'ak015fizp9ta', 'ak015fe1qm4j', 'ak015f62st0f', 'ak015f2sx6cj', 'ak015exuwagg', 'ak015exhrj0z', 'ak015eu6mocn', 'ak015ep7zz11', 'ak015eh04dck', 'ak015edn8dwc', 'ak015eaq6loc', 'ak015e22qkmc', 'ak015dlwtj7d', 'ak015dgxwogx', 'ak015df71yyh', 'ak015d8myx51', 'ak015d5dh8d4', 'ak015cou85ad', 'ak015clizfyq', 'ak015cgkcr2t', 'ak015c9yd2l3', 'ak015c87zkfv', 'ak015c4zkwr4', 'ak015btexyz0', 'ak015bq3ruqk', 'ak015bhu4fb9', 'ak015bg6lplp', 'ak015begw8pm', 'ak015b9k4aqx', 'ak015b1aohkw', 'ak015ardhotp', 'ak015ae57627', 'ak015achmqw9', 'ak0159vkqz8w', 'ak0159qzn1s8', 'ak0159pcb8u7', 'ak0159ipdx7a', 'ak0159fevf1d', 'ak0159c3t55r', 'ak01593tv00g', 'ak015926kzum', 'ak01590j25ks', 'ak0158vkg21w', 'ak0158oy0oj3', 'ak0158a0we34', 'ak01583gmkhc', 'ak0157ygktuw', 'ak0156usijy5', 'ak0156avf5px', 'ak0155fj51ix', 'ak0154lrgdy3', 'ak0154gsw7n2', 'ak0154dhtrwz', 'ak0152xvoibm', 'ak0152m9pe2s', 'ak0151vt2kxu', 'ak0151f9p9e5', 'ak01515choxm', 'ak015z2wwq8', 'ak015vf9t05', 'ak015s46nes', 'ak015gjecvt', 'ak014f2ubbs0', 'ak014f2g9ye8', 'ak014dnkb61q', 'ak014d5dfdsm', 'ak0146usu3yg', 'ak01458x0si6', 'ak0141p7hjfa', 'ak0141ikv97e', 'ak0141flljop', 'ak013ghpypm7', 'ak013dnkb3jo', 'ak013cx3r1j7', 'ak013c9y8usq', 'ak013at1abyp', 'ak0139qznfhi', 'ak0137rvjanb', 'ak0137l9fh7y', 'ak0136y3wbl1', 'ak0136kvln9g', 'ak0136fx704q', 'ak0134p2jcck', 'ak01341un6j5', 'ak0133valg7m', 'ak0132svvx2q', 'ak0132fnl9bc', 'ak0132dzz4ri', 'ak013242t91p', 'ak0131qujot9', 'ak013121eb6w', 'ak013vgmuh3', 'ak013s472lh', 'ak013ew0u0m', 'ak012gpzpsvu', 'ak012fr9du2q', 'ak012fmariyj', 'ak012ep86tli', 'ak012e43rmy8', 'ak012dxhknkx', 'ak012ddn4a1t', 'ak012cou8l54', 'ak012c1b0qfz', 'ak012bq3t26q', 'ak01286qvmq5', 'ak012853tfcj', 'ak0128058r6k', 'ak0127mwxj2s', 'ak0127bc6is9', 'ak01262m9jfn', 'ak0125w2hq4c', 'ak0125nt39pg', 'ak0124lrl4z3', 'ak0124gsvtsu', 'ak0122r89krx', 'ak0122km5uvd', 'ak0122fnlrsc', 'ak0121k8a4u8', 'ak019gg2ha2l', 'ak019da9v096', 'ak019akrdner', 'ak0198qlzw0y', 'ak0195xq4lma', 'ak019528l3mh', 'ak0191xgm3i2', 'ak0199xbirr', 'ak0182649r78', 'ak017gb3zzk7', 'ak017ew85y90', 'ak01743krbh2', 'ak0173ylxr8d', 'ak01734gkp9s', 'ak0172hb5hdl', 'ak017291cq01', 'ak0171p7ph1z', 'ak0171ikzj51']


#event_list = ['ak018aap2cqu'] #kaktovik mag6 mainshock
#event_list = ['ak014azdbz0f'] #test events
#run_name = input("Run Name (for output file names):") #YOU CAN CHANGE THIS TO FOLDER NAME MAYBE. 
path = input("Enter the path of your file: ")
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
array_code = input("Array Code (bc, bm, il, im):")

filter_type = "bandpass" # waveform filtering
freqmin = 2 # waveform & array processing filtering
freqmax = 4 # waveform & array processing filtering
ts_full = 10 #for full waveform P - ts_full (in seconds)
te_full = 10 #for full waveform P + te_full (in seconds)
ts_win = 1 # for trimmed waveform P pick window +/- (in seconds)
te_win = 4 # for trimmed waveform P pick window +/- (in seconds)
win_len = te_win+ts_win #should be x2 of t_win in seconds
ttIM = 45.5       # 317 km (6.97 km/s)
ttBM = 44.1	      # 299 km (6.78 km/s)
ttBC = 48.7       # 345 km (7.08 km/s)
ttIL =  5.5


# 0.5   2
#   1   4
#   1   8


file1 = open(path + array_code + "_obspy_processing.out","w")
file1.write("EVENT_ID"+"\t"+" AN"+"\t"+"DATE"+"\t"+"   TIME"+"\t"+"    MAG"+"\t"+"LAT"+"\t"+"   LON"+"\t"+"     DP"+"\t"+" DIST"+"\t"+" TBAZ"+"\t"+"MBAZ"+"\t"+"EBAZ"+" "+"RPW"+"  "+"SLW"+"\t"+"PICK"+"  "+"NOS"+"\t"+"TYPE"+"\n")


#plot waveforms wrt distance not alphabetically. 
#maybe work with better bmap options, if you have time


for e, lab in enumerate(event_id):
    fig1 = plt.figure(figsize=(20,8))
    ax1 = fig1.add_subplot(151)
    m = Basemap(resolution='l', projection='aea',
                llcrnrlon=190, llcrnrlat=50,
                urcrnrlon=235, urcrnrlat=72,
                lon_0=212.5, lat_0=61, ax=ax1)
    m.drawcoastlines()
    #m.drawcountries()
    #m.drawstates()
    m.fillcontinents(color='antiquewhite', lake_color='tab:blue')
    m.drawparallels(np.arange(-80., 81., 10.)) # draw parallels and meridians.
    m.drawmeridians(np.arange(-180., 181., 10.))
    m.drawmapboundary(fill_color='lightblue')
    fig1.bmap = m
    time_array = np.empty((0, 100))
    st_lat = np.empty((0, 100))
    st_lon = np.empty((0, 100))
    st_nam = np.empty((0, 100))
    cat = read_events(event_id[e])
    evmag = (cat[0].magnitudes[0].mag)
    evot = (cat[0].origins[0].time)
    evlat = cat[0].origins[0].latitude
    evlon= cat[0].origins[0].longitude
    evdep= (cat[0].origins[0].depth) / 1000 #convert to km
    evtype= cat[0].event_type
    print('##########################',event_list[e],'##########################')
    if cat[0].origins[0].evaluation_mode == "automatic":
       print("This event is not yet revised")
       continue
    event_date = cat[0].origins[0].time.date
    event_time = cat[0].origins[0].time.time.replace(microsecond=0)#TRICK TO CUT MILLISECONDS = ADD .replace(microsecond=0)
    cat.plot(fig=fig1, show=False, title="", color="date", colorbar=False)
    ax1.annotate(("{}"*5).format("Event time: ", event_date," ",event_time,"\n"), xy=(0, (1.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:8}{1:6.3f}{2:2}{3:7.3f}{4:2}{5:3.0f}{6:3}").format("Origin: ", evlat,"\u00b0"" ", evlon,"\u00b0"" ", evdep," km"), xy=(0, (1.05)), xycoords='axes fraction', fontsize=11)
    #ARRAY PICKS FROM QUAKEML
    picks = cat[0].picks
    arrivals = cat[0].origins[0].arrivals
    # REMOVED LARGE SECTION SCANNING OVER PHASE PICKS .....
  
    # REMOVED LARGE TAUP SECTION ...

    
    #WAVEFORMS FROM IRIS
    print('========= array code: ',array_code)     ################
    stn_pick = "man"
    if array_code == "bc":
       array_name = "Beaver Creek"
       t = evot + ttBC
       inventory = client_wm.get_stations(network="IM", station="BC*")
       try: #FOR THE CASES WHEN THERE ARE NO WAVEFORMS
          st = client_wm.get_waveforms("IM", "BC*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "bm":
       array_name = "Burnt Mountain"
       t = evot + ttBM
       inventory = client_wm.get_stations(network="IM", station="BM*")
       try:
          st = client_wm.get_waveforms("IM", "BM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "il":
       array_name = "Eielson"
       t = evot + ttIL
       inventory = client_wm.get_stations(network="IM", station="IL*")
       try:
          st = client_wm.get_waveforms("IM", "IL*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "im":
       array_name = "Indian Mountain"
       t = evot + ttIM
       inventory = client_wm.get_stations(network="IM", station="IM*")
       try:
          st = client_wm.get_waveforms("IM", "IM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    st.detrend(type='simple')
    st.taper(max_percentage=0.05, type='cosine')
    st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
    #PLOT STATIONS
    inventory.plot(fig=fig1, show=False)
    try:
       y_min = np.nanmin(st[:]) #for y axis to share same min and max values within all wfs
       y_max = np.nanmax(st[:])
    except NotImplementedError: #not sure why there is an error like this but this is the way around:
        print('To ambiguous, therefore not implemented, skipping this event')
        continue
    #PLOT WAVEFORMS
    nos = len(st)
    for s in range(nos):
        st[s].stats.coordinates = AttribDict({
           'latitude': inventory[0][s].latitude,
           'elevation': inventory[0][s].elevation,
           'longitude': inventory[0][s].longitude})
        st_lat = np.append(st_lat, [inventory[0][s].latitude])
        st_lon = np.append(st_lon, [inventory[0][s].longitude])
        st_nam = np.append(st_nam, [st[s].stats.station])
        ax2 = fig1.add_subplot(nos, 4, ((4*s)+2))
        wf_all = st[s]
        ax2.plot(wf_all.times("matplotlib"), wf_all.data, "k-", linewidth=0.3)
        text = (st[s].stats.station)
        ax2.set_ylim(ymin=y_min, ymax=y_max)
        ax2.xaxis_date()
        ax2.axvline(date2num((t-ts_win).datetime), lw=0.8, c='darkblue', ls='--', label='time win.')
        ax2.axvline(date2num((t+te_win).datetime), lw=0.8, c='darkblue', ls='--')
        ax2.axvline(date2num(t.datetime), lw=0.8, c='darkred', label='P pick')
        ax2.text(0.10, 0.95, text, transform=ax2.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')
        st_trim = st[s].trim(t - (ts_win+0.1), t + (te_win+0.1)) #trim is done inside the loop, because otherwise st changes and whole wf cannot be plotted
        wf_trim = st_trim.taper(max_percentage=0.1, type='cosine')
        ax3 = fig1.add_subplot(nos, 4, ((4*s)+3))
        #ax3.set_ylim(ymin=y_min/10, ymax=y_max/10) #until we figure out a better way to do this, it seems to work. or the following.
#         if y_max >= 1000: 
#            ax3.set_ylim(ymin=y_min/10, ymax=y_max/10)
#         else:
#            ax3.set_ylim(ymin=y_min, ymax=y_max)        
        ax3.plot(wf_trim.times("matplotlib"), wf_trim.data, "k-", linewidth=0.3)
        ax3.xaxis_date()
        ax3.axvline(date2num((t-ts_win).datetime), lw=0.8, c='darkblue', ls='--', label='time win.')
        ax3.axvline(date2num((t+te_win).datetime), lw=0.8, c='darkblue', ls='--')
        ax3.axvline(date2num(t.datetime), lw=0.8, c='darkred')
        ax3.text(0.10, 0.95, text, transform=ax3.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.legend(loc='lower left', fontsize=6)
    ax2.annotate(("{}"*5).format("Full waveform: P - ",ts_full, " sec. to P + ",te_full, " sec."), xy=(0, (nos+0.1)), xycoords='axes fraction', fontsize=11)
    fig1.autofmt_xdate()
    if array_code == "IL":
       plt.text(date2num(t.datetime)-0.000002,0,'   P',rotation=90, color='darkred')
    else:
       plt.text(date2num(t.datetime)-0.000002,0,'P pick   ',rotation=90, color='darkred')
    ax3.annotate(("{}"*5).format("Trimmed waveform: P - ",ts_win, " sec. to P + ",te_win, " sec."), xy=(0, (nos+0.1)), xycoords='axes fraction', fontsize=11)
    ax3.legend(loc='lower left', fontsize=6)
    #GREAT CIRCLE BACKAZIMUTH & DISTANCE CALCULATION
    stalat = np.mean(st_lat)
    stalon = np.mean(st_lon)
    client = Client()
    gc_distaz = client.distaz(stalat=stalat, stalon=stalon, evtlat=evlat, evtlon=evlon)
    gc_dist = gc_distaz['distance']
    gc_baz = gc_distaz['backazimuth']
    #ax3.axvline(date2num(cal_pt.datetime), lw=0.8, c='cyan') #was used to plot taup pick. 
    stime = t - ts_win
    etime = t + te_win
    kwargs = dict(
       # slowness grid: X min, X max, Y min, Y max, Slow Step
       sll_x=-0.2, slm_x=0.2, sll_y=-0.2, slm_y=0.2, sl_s=0.002,
       # sliding window properties
       win_len=win_len, win_frac=0.05,
       # frequency properties
       frqlow=freqmin, frqhigh=freqmax, prewhiten=0,
       # restrict output
       semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
       stime=stime, etime=etime
    )
    out = array_processing(st, **kwargs)
    cl_rlp = out[:, 1] #calculated relative power
    cl_abp = out[:, 2] #calculated absolute power
    cl_baz = out[:, 3] #calculated backazimuth
    cl_slw = out[:, 4] #calculated slowness
#OBSPY ARRAY PROCESSING PLOT #2 POLAR PLOT
    cmap = obspy_sequential #colormaps
    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 0.3, N2 + 1)
    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    # transform to radian
    baz_edges = np.radians(baz_edges)
    # add polar and colorbar axes
    cax4 = fig1.add_axes([0.9, 0.20, 0.035, 0.15])
    ax4 = fig1.add_axes([0.65, 0.35, 0.40, 0.4], polar=True)
    ax4.set_theta_direction(-1)
    ax4.set_theta_zero_location("N")
    ax4.set_thetagrids(np.arange(0, 360, 20), labels=np.arange(0, 360, 20))
    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])
    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax4.bar(x=(i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2),
                     color=cmap(row / hist.max()))
    # set slowness limits
    ax4.set_ylim(0, 0.3)
#    ax4.set_title(("{}"*5).format("Beaver Creek "))
    [i.set_color('grey') for i in ax4.get_yticklabels()]
    ColorbarBase(cax4, cmap=cmap,
                norm=Normalize(vmin=hist.min(), vmax=hist.max()))
#OBSPY ARRAY PROCESSING PLOT #1
#    labels = ['rel_power', 'abs_power', 'baz', 'slowness']
#     xlocator = mdates.AutoDateLocator()
#     for i, lab in enumerate(labels):
#         ax4 = fig1.add_subplot(4, 4, (i + 1)*4)
#         ax4.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
#                    edgecolors='none', cmap=obspy_sequential)
#         ax4.set_ylabel(lab)
#         ax4.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
# #        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #no need for now.
#         ax4.yaxis.set_label_position("right")
#         ax4.xaxis.set_major_locator(xlocator)
#         ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     ax4.annotate(("{}"*1).format("Array Processing with Obspy"), xy=(0, 4.08), xycoords='axes fraction', fontsize=11)
    err_baz = gc_baz - cl_baz
    if err_baz > 180:
       err_baz = err_baz - 360
    else:
       err_baz = err_baz
    #print('------>  ',gc_baz,float(cl_baz),err_baz) ######
    fig1.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    file1.write("{0:12} {1:2} {2:} {3:} {4:3.1f} {5:6.3f} {6:7.3f} {7:3.0f} {8:6.3f} {9:6.2f} {10:6.2f} {11:7.2f} {12:4.2f} {13:5.3f} {14:4} {15:2}     {16:10}".format(event_list[e],array_code,event_date,event_time,evmag,evlat,evlon,evdep,float(gc_dist),float(gc_baz),float(cl_baz),float(err_baz),float(cl_rlp),float(cl_slw),stn_pick,nos,evtype))
    file1.write("\n")
    #TEXT TO APPEND
    ax1.annotate(("{}"*7).format("Filter type: ",filter_type, " ", freqmin, " - ", freqmax, " Hz" ), xy=(0, (-0.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*2).format("P pick is from: ",stn_pick), xy=(0, (-0.2)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*3).format("Time window length: ",win_len, " sec."), xy=(0, (-0.3)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:10}{1:6.2f}{2:1}").format("True BAZ: ",float(gc_baz),"\u00b0"), xy=(0, (-0.4)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:14}{1:4.2f}{2:1}").format("Apparent BAZ: ",float(np.mean(cl_baz)),"\u00b0"), xy=(0, (-0.5)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:5.3f}{2:1}").format("Apparent slowness: ",np.mean(cl_slw)," s/km"), xy=(0, (-0.6)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:4.2f}").format("Relative power: ",np.mean(cl_rlp)), xy=(0, (-0.7)), xycoords='axes fraction', fontsize=11)
    fig1.suptitle(("{}"*4).format("Event Name: ", event_list[e][2:], ", Array: ", array_name), y=1.04,fontweight='bold')
    fig1.savefig(path + event_list[e] + "_" + array_code + '.pdf', bbox_inches='tight')
#    fig1.savefig(run_name + "_" + event_list[e] + "_" + array_code + '_base.pdf', bbox_inches='tight')
file1.close()
    