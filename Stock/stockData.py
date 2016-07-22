import tushare as ts

def download(sockNum):
    d = ts.get_hist_data(sockNum)
    d.to_csv(sockNum+".data",sep = "\t")
    out = open(sockNum,"w")
    out.write(d.__str__())
    out.close()