# Yöntem: final modelin seçimi

Final model, DS2 sonucuna bakılmadan kilitlenmiştir. Seçim ile test iki ayrı adımdır.

| Adım | Veri | Ne için kullanılır |
|---|---|---|
| Model seçimi | Yalnızca DS1, hasta bazlı çapraz doğrulama | Model, füzyon, dengesizlik yöntemi |
| Son eğitim | Yalnızca DS1 | Kilitlenen modelin yeniden eğitilmesi |
| Final test | Yalnızca DS2, bir kez | Kilitlenen modelin bildirilen test sonucu |

DS2 skoru model, füzyon, dengesizlik yöntemi veya tohum seçmek için kullanılmaz. Test sonucuna bakıp aday değiştirilmez.

## DS1 üzerinde hasta bazlı çapraz doğrulama

Seçim, MIT-BIH DS1 kayıtları üzerinde 4 katlı çapraz doğrulama ile yapılır. Kayıt 124 bu kümenin içindedir ve katlara dahildir. Kat ataması tohum 42 ile yapılır ve bütün adaylar aynı katları kullanır.

Bir kayıt bir katta ya eğitimde ya doğrulamadadır. Aynı kaydın vuruşları iki tarafa birden düşmez. DS2 kayıtları hiçbir katın eğitim veya doğrulama kümesine girmez.

Birincil ölçüt macro-F1’dir. Doğruluk ikincildir. Adaylar, katlar ve tohumlar üzerinden macro-F1 ortalamasına göre sıralanır. İki adayın ortalaması 0,01 içindeyse daha düşük standart sapmalı aday tutulur. Standart sapma örneklem standart sapmasıdır. F ve Q sınıfları ortalamadan çıkarılmaz.

Karşılaştırma tohumları 42, 43 ve 44’tür. Bütün adaylar aynı bütçeyi paylaşır: 8 epoch, yığın boyutu 128, öğrenme hızı 0,001.

## Karşılaştırılan üç kurulum

Dengesizlik yöntemi önce, yalnızca DS1 katlarında ve tohum 42 ile, 1D + 2D + klinik + attention modeli üzerinde taranır. Bu tarama DS2 kullanmaz.

Ardından üç kurulum aynı DS1 katlarında ve üç tohumda karşılaştırılır:

1. 2D spektrogram + klinik özellikler + concat, referans dengesizlik ayarı (dengeli sınıf ağırlığı, cross-entropy, örnekleyici yok)
2. 1D + 2D + klinik özellikler + attention, aynı referans ayar
3. 1D + 2D + klinik özellikler + attention, DS1 taramasının seçtiği dengesizlik ayarı

Bu taramada kazanan referans ayarın kendisi olduğu için üçüncü kurulum ikinci ile aynı kalmıştır. Seçim kuralı 1. kurulumu final model yapmıştır: 2D spektrogram + klinik özellikler + concat.

## DS1 üzerinde yeniden eğitim

Seçim kilitlendikten sonra bu model DS1 üzerinde yeniden eğitilir. DS1 içinden, hasta bazlı ve tohum 42 ile ayrılan %20’lik dilim yalnızca en iyi kontrol noktasını saklamak için kullanılır. Bu dilim DS2 değildir ve ikinci bir test kümesi değildir.

## DS2 üzerinde final test

Yeniden eğitim bittikten sonra DS2 bir kez değerlendirilir. Bu değerlendirme seçim tablosuna geri yazılmaz. Dış doğrulama ve açıklanabilirlik çıktıları da aynı kilitli kontrol noktasından üretilir; onlar da model seçimini değiştirmez.
