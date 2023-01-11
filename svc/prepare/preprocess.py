import random

if __name__ == "__main__":

    alls = []
    fo = open("./filelists/vc_file.txt", "r+")
    while True:
        try:
            message = fo.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        alls.append(message)
    fo.close()

    valids = alls[:1]
    trains = alls[1:]

    random.shuffle(trains)

    fw = open("./filelists/singing_valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()

    fw = open("./filelists/singing_train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
