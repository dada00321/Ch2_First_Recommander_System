import os
import json
import random
import math
import time

class FirstRec():
    def __init__(self, file_path, seed, k_users, n_items, test_num=17770):
        self.file_path = file_path
        self.seed = seed
        self.k_users = k_users
        self.n_items = n_items
        self.test_num = test_num
        self.users_1k = self.select_1k_users()
        self.train, self.test = self.load_and_split_data()

    #--------------------------------------------------------
    def select_1k_users(self):
        if os.path.exists("res/train.json") and os.path.exists("res/test.json"):
            return list()
        else:
            print("正在隨機抽取 1000 名用戶...")
            users = self.__collect_non_repetitive_users()
            #print(f"# users: {len(users)}")
            #print(users[:10])
            users_1k = random.sample(users, 1000)
            #print(users_1k)
            return users_1k
    def __collect_non_repetitive_users(self):
        users = set()
        progress_base = self.test_num // 10
        for i, file in enumerate(os.listdir(self.file_path)[:self.test_num]):
            path = f"{self.file_path}/{file}"
            #print(path)
            if not (i+1) % progress_base: print("進度: {:.1f} %".format((i+1)*100/self.test_num))
            with open(path, "r") as fp:
                for record in fp.readlines():
                    users.add(record.split(",")[0])
        print("1000名用戶抽樣成功！\n")
        return list(users)
    #--------------------------------------------------------
    def load_and_split_data(self):
        train = dict()
        test = dict()
        if os.path.exists("res/train.json") and os.path.exists("res/test.json"):
            print("找到先前保存的 dataset, 正在將其載入...")
            train = json.load(open("res/train.json"))
            test = json.load(open("res/test.json"))
            print("dataset載入完成！\n")
        else:
            random.seed(self.seed) # 確保在不同試驗下，隨機度皆保持一致
            repetitive_count = 0
            print("正在拆分 dataset ...")
            progress_base = self.test_num // 10
            for i, file in enumerate(os.listdir(self.file_path)[:self.test_num]):
                movID = str(int(file[6:].split(".")[0]))
                path = f"{self.file_path}/{file}"
                #print(path)
                if not (i+1) % progress_base: print("進度: {:.1f} %".format((i+1)*100/self.test_num))
                with open(path, "r") as fp:
                    data = fp.readlines()
                for record in data:
                    usrID, rating, _ = record.split(",")
                    # 若該用戶是1000個隨機用戶樣本中的其中一員，加入training set 或 test set
                    if usrID in self.users_1k:
                        if random.randint(1, 50) > 1: # 98% 機率加入training set
                            train.setdefault(usrID, {})[movID] = int(rating)
                        else: # 2% 機率加入test set
                            if usrID in train.keys(): repetitive_count += 1
                            test.setdefault(usrID, {})[movID] = int(rating)
            print("dataset 拆分完成！\n")
            print(" repetitive_count:", repetitive_count, "\n") # 4634 
            
            #print("training set:", train, sep="\n")
            #print("test set:", test, sep="\n")
            
            if not os.path.exists("res"): os.mkdir("res")
            print("正在保存 dataset 為 JSON 檔案...")
            json.dump(train, open("res/train.json","w"))
            json.dump(test, open("res/test.json","w"))
            print("JSON檔保存成功！\n")
            
        print(f" # movies(sample): {self.test_num}")
        users_1 = train.keys()
        users_2 = test.keys()
        print(f" # users(sample): {len(users_1)+len(users_2)}")
        print(f"    ∟ training set: {len(users_1)} users")
        print(f"    ∟ test set: {len(users_2)} users\n")
        '''
        # movies(sample): 17770
        # users(sample): 1765
           ∟ training set: 1000 users
           ∟ test set: 765 users
        '''
        return train, test
    #--------------------------------------------------------
    """ 計算 Pearson correlation coefficient
    rating_1: 用戶1的評分紀錄 E.g. {"movie_1": 3, "movie_2": 5}
    rating_2: 用戶2的評分紀錄 E.g. {"movie_1": 4, "movie_2": 3}
    """
    def pearson(self,rating1,rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        num = 0
        for key in rating1.keys():
            if key in rating2.keys():
                num += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += math.pow(x,2)
                sum_y2 += math.pow(y,2)
        if num == 0:
            return  0
        # Pearson相關係數分母
        denominator = math.sqrt( sum_x2 - math.pow(sum_x,2) / num) * math.sqrt( sum_y2 - math.pow(sum_y,2) / num )
        if denominator == 0:
            return  0
        else:
            return ( sum_xy - ( sum_x * sum_y ) / num ) / denominator
    
    def recommand(self, usrID):
        movies_of_ursID = self.train[usrID].keys()
        
        neighbor_users = dict()
        for user in self.train.keys():
            if user != usrID:
                pearson_r = self.pearson(self.train[user], self.train[usrID])
                neighbor_users[user] = pearson_r
                print(f"user:{user} pearson_r:{pearson_r}")
            
        # 找出前 k_users 位相似的用戶
        newNU = sorted(neighbor_users.items(), key=lambda x:x[1], reverse=True)[:self.k_users]
        #print(newNU)
        rec_movies = set()
        '''
        for nu in newNU:
            if len(rec_movies) >= self.n_items: break
            user = nu[0] # 0: user / 1: pearson's correlation coefficient
            #print("user: ", user)
            movies = self.train[user]
            for mov in movies:
                if len(rec_movies) < self.n_items and :
                    rec_movies.add(mov)
            #print(*(mov for mov in movies), end="\n\n")
        print(rec_movies)
        '''
if __name__ == "__main__":
    """
    <population>
     # movies(tot.): 17770
     # users(tot.): 2649429
    --------------------------------------
    <sample>
     # movies(tot.): # test_num
     # users: 1000
    """
    file_path = "data/training set"
    seed = 30
    k_users = 15 # 最相鄰(相似)的前 15 用戶
    n_items = 20 # 為每位用戶推薦的電影數
    ''' <Observation>
        #E.g., test_num:        100  ->  500 
               repetitive_count: 12  ->  145
                                 12% ->  29%
    '''
    test_num = 50 # 17770部電影中的測試資料筆數
    #rec = FirstRec(file_path, seed, k_users, n_items, test_num) # for test 
    rec = FirstRec(file_path, seed, k_users, n_items)
    
    # 計算用户 userID_1 和 userID_2 的皮爾森相關係數
    #userID = "7106"; userID_2 = "2317930" 
    userID = "195100"; userID_2 = "1547579" 
    
    r = rec.pearson(rec.train[userID], rec.train[userID_2])
    print("用戶\'{}\' 和 用戶\'{}\'的皮爾森相關係數 r = {:.2f}\n".format(userID, userID_2, r))
    """
    A = set(rec.train[userID].keys())
    B = set(rec.train[userID_2].keys())
    A_and_B = A.intersection(B)
    print("A_and_B:", A_and_B, end="\n\n")

    print("user_1:")
    u = []
    for k, v in rec.train[userID].items():
        if k in A_and_B:
            u.append(f"{k}: {v}")
    print(sorted(u), end="\n\n")
    print("user_2:")
    for k, v in rec.train[userID_2].items():
        if k in A_and_B:
            u.append(f"{k}: {v}")
    print(sorted(u), end="\n\n")
    """
    userID = "301766"
    print(f"為用戶{userID}推薦的電影:")
    rec.recommand("301766")