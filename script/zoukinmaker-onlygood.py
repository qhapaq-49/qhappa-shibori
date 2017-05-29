# coding: UTF-8
import sys

# python zoukinmaker.py filename minimum
#標準入力の引数読み込み
args = sys.argv
print args

# 逆転模様の棋譜についても逆転し始めた点から使う
# 勝った側の点数が悪い局面は評価値が正確でないのか入れると弱くなる傾向にある（TODO：高精度測定）
# しかし、優位を保持して勝った局面だけ集めると、中終盤でイーブンな棋譜が不足する（かもしれない）
usegyaku = True

minimumv = int(args[1])
mixparam = float(args[2])

linekif = "aiee" # 1行を文字列として読み込む
linevalue = "aba"
lineresult = "otassyade"


for fn in range(len(args)-3):
    f = open(args[fn+3])
    outkif = open(args[fn+3]+".out.sfen","w")
    outvalue = open(args[fn+3]+".out.txt","w")

    print(args[fn+3] + "start")
    linekif = f.readline()
    linevalues = f.readline().split(' ')
    lineresults = f.readline().split(' ')

    while len(lineresults) > 1:
        #drawは除外
        #print(lineresults)
        if int(lineresults[1]) != 3 :
            if usegyaku :
                # この手数までは評価値を0とする
                invalid = 0
            else:
                # 途中て逆転模様とかになってたらFalseになる
                isUse = True
                
            linevalues = filter(lambda s:(s != "" and s !="\n" and s!="\r\n"), linevalues)
            #print linevalues
            outv = ""
            convkey = int(0)
            if int(lineresults[0])==int(lineresults[1])-1 :
                convkey = 1 # 先手勝ち
            else:
                convkey = 0 #後手勝ち
            stop = False
            tps = []
            for i in range(len(linevalues)):
                #tp = 0
                if i%2 == convkey:
                    #tp = 0
                    tps.append(0)
                else:
                    tp = int(linevalues[i])
                    if tp < minimumv and i > 16:
                        if usegyaku:
                            invalid = i
                        else:
                            isUse = False
                    if convkey == 0 :
                        tp = -tp
                        #mixing
                        #tp = oldtp
                    tps.append(tp)
                #outv = outv + str(tp) + " "
            #print outv
            if usegyaku:
                isUsefinal = True
            else:
                isUsefinal = isUse
            
            if isUsefinal == True:
                outkif.write(linekif)
                #outvalue.write(outv+"\n")
                #tpsを素のスコアから数手先まで混ぜたスコアに変換する(pokerのAIの戦略を参照)
                for i in range(len(tps)):
                    tpscale = 1.0
                    tpscalesum = 0.0
                    outtp = 0.0
                    if tps[i] != 0 and i >= invalid:
                        for j in range((len(tps)-i)/2):
                            sv = tps[i+j*2]
                            #mateは2017としておく。特に意味はないかも
                            if sv < -2017 and j!= 0:
                                sv = -2017
                            if sv > 2017 and j!= 0:
                                sv = 2017
                            outtp += tpscale * sv
                            tpscalesum += tpscale
                            tpscale = tpscale * mixparam
                        outtp = outtp / (tpscalesum+0.000001)
                    outv += str(int(outtp)) + " "
                outvalue.write(outv+"\n")
        linekif = f.readline()
        linevalues = f.readline().split(' ')
        lineresults = f.readline().split(' ')
    print(args[fn+3]+" done")
    f.close()
    outkif.close()
    outvalue.close()
