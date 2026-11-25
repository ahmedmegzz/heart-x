# Final model

Bu çalışmanın final modeli **2D spektrogram + klinik özellikler + concat** yapısıdır. Kod adı `2d_clinical`, füzyon `concat`, kontrol noktası `outputs/checkpoints/heartx_final.pt`.

Model iki koldan oluşur. Birinci kol, EKG vuruşunun spektrogramını 2D evrişimli ağ ile işler. İkinci kol, aynı vuruşun klinik özelliklerini (RR aralıkları ve dalga biçimi özetleri) bir ileri beslemeli ağ ile işler. İki kolun öznitelik vektörleri concat ile uç uca birleştirilir ve sınıflandırıcıya verilir. 1D sinyal kolu bu final modelde yoktur. Concat, öğrenilmiş bir attention ağırlığı değildir; iki vektörü birleştirir.

## Neden bu yapı seçildi

Seçim adımları yöntem bölümünde yazılıdır (`yontem_model_secimi.md`). Karar yalnızca DS1 üzerindeki hasta bazlı çapraz doğrulamaya dayanır. DS2 bu karardan sonra bir kez test için kullanılmıştır.

Seçim yalnızca DS1 üzerinde, hasta bazlı 4 katlı çapraz doğrulama ile yapıldı. Tohumlar 42, 43 ve 44, eğitim 8 epoch, birincil ölçüt macro-F1’dir. DS2 seçime katılmadı. Kayıt 124 DS1 içinde kaldı.

Aynı katlarda ve aynı tohumlarda üç kurulum karşılaştırıldı:

1. 2D spektrogram + klinik özellikler + concat
2. 1D + 2D + klinik özellikler + attention
3. 2. kurulum + attention modeli üzerinde en iyi dengesizlik yöntemi

Attention modeli üzerindeki dengesizlik taramasında en iyi ayar, referans ayarın kendisi oldu: dengeli sınıf ağırlığı, cross-entropy, örnekleyici yok. Bu nedenle üçüncü kurulum ikinci ile aynıdır ve ayrı bir model üretmemiştir.

DS1 çapraz doğrulama sonuçları:

| Kurulum | macro-F1 | Doğruluk |
|---|---|---|
| 2D + klinik + concat | 0,4291 ± 0,0779 | 0,9440 ± 0,0228 |
| 1D + 2D + klinik + attention | 0,4224 ± 0,0847 | 0,9319 ± 0,0414 |

Başlangıçta düşünülen üçlü attention yapısı daha iyi sonuç vermedi. Macro-F1 ortalaması daha düşük, standart sapması daha yüksektir. Seçim kuralı en yüksek ortalama macro-F1’i alır; fark 0,01 içindeyse daha düşük standart sapmalı aday tutulur. Her iki ölçütte de 2D + klinik + concat öndedir. Final model bu sonuç nedeniyle kilitlendi. Attention yapısı final model olarak seçilmedi.

Kilitlenen model DS1 üzerinde yeniden eğitildi. DS2 bu eğitimden sonra yalnızca bir kez test edildi.
