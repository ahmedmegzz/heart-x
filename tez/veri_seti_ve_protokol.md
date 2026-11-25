# Veri seti ve değerlendirme protokolü

Bu bölüm, MIT-BIH Aritmi Veritabanı üzerinde kullanılan kayıt ayrımını tanımlar. Model seçimi ve test bu ayrıma bağlıdır. Yeni bir model veya yeni bir deney bu bölümün konusu değildir.

## MIT-BIH Aritmi Veritabanı

Çalışmanın ana veri kaynağı MIT-BIH Aritmi Veritabanı’dır. Veritabanında 48 kayıt bulunur. Vuruşlar AAMI eşlemesine göre beş sınıfa indirgenir: N (normal ve ilişkili vuruşlar), S (supraventriküler ektopik), V (ventriküler ektopik), F (füzyon) ve Q (sınıflandırılamayan, pacemaker vuruşları dahil).

## Pacemaker kayıtlarının çıkarılması

102, 104, 107 ve 217 numaralı kayıtlar kalp pili (paced) vuruşları içerdiği için çalışmanın tamamından çıkarılmıştır. Bu dört kayıt DS1’e de DS2’ye de konmamıştır. Çıkarma işleminden sonra değerlendirme, kalan 44 kayıt üzerinden yapılır.

Pacemaker kayıtları çıktıktan sonra Q sınıfı çok seyrek kalır. Bu durum, Q üzerindeki düşük başarımın veri sınırlılığı olarak ele alınmasının gerekçesidir. Q sınıfı etiket olarak korunur; düşük sonuç gizlenmez.

## DS1–DS2 ayrımı

Kalan 44 kayıt, de Chazal ve arkadaşlarının hasta bazlı (inter-patient) ayrımına göre iki kümeye bölünür. Ayrım kayıt düzeyindedir: bir kaydın vuruşları yalnızca bir kümede yer alır. İki küme örtüşmez.

**DS1** (22 kayıt) model seçimi, eğitim ve doğrulama için kullanılır:

101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230.

**DS2** (22 kayıt) yalnızca test kümesidir:

100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.

124 numaralı kayıt DS1’in içindedir ve DS2’de yer almaz. Hasta bazlı model seçiminde ve DS1 üzerindeki son eğitimde kullanılır.

## DS2’nin rolü

DS2, modelin, füzyon yönteminin, dengesizlik yönteminin veya tohumun seçiminde kullanılmaz. Konfigürasyon yalnızca DS1 üzerindeki hasta bazlı doğrulama ile kilitlenir. Kilitlenen model DS1 üzerinde yeniden eğitilir ve DS2 üzerinde bir kez test edilir.

## DS1 içindeki eğitim ve doğrulama

Eğitim ve doğrulama ayrımı yalnızca DS1 içinde ve hasta (kayıt) bazında yapılır. Aynı kayda ait vuruşlar hem eğitim hem doğrulama kümesine düşmez. DS2 bu ayrımın dışındadır.
