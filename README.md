# Plot Digitizer

PDF'teki grafiklerden ve **grafik görsellerinden (PNG, JPG, BMP, TIFF, WebP)** **lookup tablosu** üretir. PDF açılınca içindeki tüm grafikler otomatik bulunur; sayfaya çizilmiş
(vektör) grafiklerin eğri değerleri PDF'in **vektör verisinden** (piksel ölçümü değil), PDF'e resim olarak yapıştırılmış
grafiklerin değerleri eksen kalibrasyonuyla resimden okunur. Sonra bir grafik ve bir eğri seçilir; tablo görülür, X ya da
Y yazılarak karşılığı sorgulanır, hatalı noktalar düzeltilir, CSV/PDF olarak kaydedilir ya da yazdırılır.

Python 3.11+, PySide6, NumPy, SciPy, pandas, pypdfium2, OpenCV (resim olarak gömülü grafikler ve eski araç için).

## Kullanım

1. **PDF ya da görsel aç** (`Ctrl+O` ya da sürükle-bırak). PDF'in tüm sayfalarındaki grafikler taranır (~1 sn/sayfa); bir
   görsel dosyası tek grafik olarak açılır (aşağıda "Resim olarak gömülü grafikler ve görsel dosyaları").
2. **Galeri**: her grafik görseli ve adıyla gelir; birine tıklayın.
3. **Detay**: solda grafik, sağda **eğriler ve anlamları** (legend metni + hangi eksene ait + çizgi türü) ve grafik
   notları. Bir eğri seçince (seçili eğri grafikte sarı ile işaretlenir) altında **lookup tablosu** oluşur.
   Fare grafiğin üstündeyken durum çubuğu X, Y değerini ve seçili eğrinin o X'teki değerini gösterir.
4. **X adımı**: otomatik yuvarlak bir adım (≈50 satır); değiştirilebilir, `0` = eğrinin kendi noktaları.
5. **Değer sorgula** (tablonun altında): `X = …` yazın, eğrinin Y'sini; `Y = …` yazın, eğrinin X'ini görün (eğri aynı Y'yi
   birkaç kez alıyorsa hepsi listelenir). Ondalık için virgül ya da nokta. Karşılık, grafikte artı işaretiyle gösterilir;
   `±` değer, noktaların belirsizliğinden ve eğimden hesaplanır. Eğrinin X/Y aralığı dışında değer üretilmez.
6. **Düzelt** (grafiğin altında): yanlış okunan yeri düzeltmek için **Nokta ekle**, **Nokta sil**, **Kutuyla sil**,
   **Nokta taşı** (bir noktayı tutup sürükleyin); `Ctrl+Z` / `Ctrl+Y` geri al/yinele, **Özgün hale döndür** hepsini
   siler. Düzeltilen eğri listede "düzeltildi (N)" yazar; tablo, sorgu, CSV, PDF ve yazdırma düzeltilmiş veriyi kullanır ve
   çıktıda "eğri elle düzeltildi" notu bulunur. Elle eklenen noktanın belirsizliği, tıklamanın piksel çözünürlüğüdür.
7. **CSV**, **PDF kaydet…**, **Yazdır…** (sistem yazdırma penceresi; "Microsoft Print to PDF" da seçilebilir).

Değerler arasında doğrusal interpolasyon yapılır (kesikli çizgilerde dash boşlukları dahil); eğrinin X aralığı
dışında değer üretilmez. Belirsizlik, PDF'in kendi koordinat çözünürlüğüdür (bu datasheet'te ±0,06 pt →
≈ ±1,2 mAh, ±0,0026 V) ve tablo başlığında yazılır.

### Resim olarak gömülü grafikler ve görsel dosyaları (ör. Aspilsan datasheet'i, ekran görüntüsü, tarama)

Bazı datasheet'lerde grafik çizim değil, PDF'e yapıştırılmış bir resimdir (PNG/JPG). Böyle grafikler de galeride
görünür ("görsel grafik · kalibrasyon gerekli"; adı resmin üstündeki PDF metninden alınır). **PNG, JPG, BMP, TIFF ve WebP
dosyaları da aynı yoldan** açılır: dosya tek grafik olarak gelir (adı dosya adıdır), çıktılarda sayfa numarası yazmaz.
Çerçeve otomatik aranır; bulunamazsa (çerçevesiz grafik ya da çok gürültülü resim) **Grafik alanı…** ile eksenlerin oluşturduğu
dikdörtgeni çizersiniz (bu alan da kalibrasyonla birlikte hatırlanır). Sonrası aşağıdaki adımlarla aynıdır:

1. Grafiği açın, **Kalibre et**'e basın; X ekseninde iki, Y ekseninde iki işarete (tick) tıklayın. Tıklama yakındaki tick
   çizgisine ya da ızgaraya yapışır; grafik çerçevesinin kenarları da önerilir. Sağda ikinci bir Y ekseni varsa "evet"
   deyip iki işaret daha gösterin.
2. Açılan pencereye işaretlerin değerlerini ve eksen adlarını yazın (birimi parantez içinde yazarsanız tablo başlığında
   görünür: `Gerilim (V)`); logaritmik eksen için kutuyu işaretleyin. Metin okuma (OCR) yoktur, değerleri siz verirsiniz.
3. Eğriler renklerinden **otomatik** bulunur (aynı renkte iki eğri, örn. gerilim ve sıcaklık, ayrı bulunur; sağ eksen renginde
   yazılmış eksen başlığı olan eğriler sağ eksene atanır). Legend resim olduğu için adlar `Eğri N (renk)` gelir:
   **Adı…** (`F2`) ile değiştirin. Eksik ya da siyah/gri eğri için **＋ Eğri ekle** ile eğrinin üstüne tıklayın
   (kesikli çizgiyse "Kesikli"yi işaretleyin); fazla eğri için **Eğriyi sil**; yanlış eksene atananı **Eksen ⇄** çevirir.
4. Sonrası vektör grafiklerle aynıdır: tablo, değer sorgula, düzelt, CSV/PDF/yazdır.
5. **Kalibrasyon hatırlanır:** kalibrasyonu bitirince kendiliğinden kaydedilir; aynı PDF'i (ya da adı değişmiş bir kopyasını)
   bir daha açtığınızda o grafik **sorulmadan** kalibre gelir ve eğrileri okunur (galeride "kalibrasyon kayıtlı" yazar).
   PDF, dosya içeriğinin özetiyle tanınır: taşımak ya da yeniden adlandırmak sorun olmaz; içeriği değişen (başka) bir
   PDF için yeniden sorulur. Değiştirmek için **Yeniden kalibre et** (eskisinin yerine yazılır); silmek için
   `Araçlar → Bu grafiğin kayıtlı kalibrasyonunu sil`. Kayıt yalnızca kalibrasyonu içerir; eğri adları ve elle
   düzeltmeler oturumla sınırlıdır. Dosya: `%APPDATA%\PlotDigitizer\calibrations.json` (`PLOT_DIGITIZER_DATA`
   ortam değişkeniyle başka klasöre alınabilir; silmek her şeyi unutturur). Vektör grafikler kalibrasyon gerektirmez.

Doğruluk resmin piksel çözünürlüğüyle sınırlıdır (tablo altında ± olarak yazar; Aspilsan grafiklerinde ≈ ±2 mAh, ±0,004 V).
JPEG sıkıştırması çizgi kenarlarını bulanıklaştırır ama beş eğrilik deneme grafiğinde (600 DPI PNG) bir pikselin altında
kalır: PDF'in vektör değerlerine göre ortalama sapma +0,003 V, en büyük 0,0044 V.
Şarj grafiğinde datasheet'in kendi değerleri okunur: 1400 mA şarj akımı, 4,2 V bitiş gerilimi, 140 mA kesme akımı, ≈2830 mAh.

### Nasıl okunuyor?

* Grafik çerçevesi, başlığı, eksen adları ve tick etiketleri PDF'ten bulunur; eksen kalibrasyonu tüm tick'lere
  en küçük kareler uyumuyla yapılır (lineer ya da log10).
* Eğriler: uç uca değen parçalar → sürekli eğri, aralıklı kısa parçalar → kesikli eğri, küçük kapalı şekiller →
  işaretçi serisi. Izgara, çerçeve, tick ve legend çizgileri ayıklanır. Başka eğrinin altında kalan bölümler de
  (vektör verisi orada durduğu için) tam okunur.
* Anlam: legend işareti + yanındaki metin; legend yoksa eksen rengi (mavi eksen → mavi eğri) ya da çizgi türü.
  Çok Y eksenli grafiklerde eğri renginin eksen rengiyle eşleşmesine göre doğru eksenden okunur.

### Sınırlar

* Otomatik bulunan grafikler: sayfaya çizilmiş (vektör) grafikler, **içinde çerçeveli bir grafik bulunan resimler** ve
  görsel dosyaları (PNG/JPG/BMP/TIFF/WebP). Tam sayfa taranmış PDF'lerde (sayfanın tamamı bir resim, içinde birkaç grafik)
  `Araçlar → Görselden elle sayısallaştır (gelişmiş)` (eski, her şeyi elle yapılan araç; `python app.py --manual resim.png`)
  ya da sayfayı görsel olarak kaydedip açmak gerekir.
* Resim grafiklerde: eksen değerleri elle girilir (OCR yok); ikinci Y ekseni desteklenir ama üçüncüsü yoktur; kesikli, siyah
  ve gri eğrilerle çakışan eğriler için `＋ Eğri ekle` gerekebilir; başka eğrinin altında kalan bölümü resimde göremeyiz
  (vektör grafiklerden farklı olarak); eğrilerin eksenle kesiştiği ilk birkaç piksel kalın çerçeve yüzünden okunmayabilir.
* Grafik tespiti çerçeveyi dört ince kenar çizgisinden bulur; çerçevesiz grafikler bulunamaz. Log eksenlerde
  `10^n` biçiminde (mathtext) tick etiketleri okunamaz, düz sayı etiketleri (1, 10, 100) okunur.
* Aynı renk + aynı çizgi türünde iki ayrı eğri tek eğri olarak birleşir.

## Kurulum, çalıştırma, test

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py                                  # arayüz
.\.venv\Scripts\python app.py samples\INR18650P28A-V1-80093.pdf
.\.venv\Scripts\python -m pytest                               # testler
.\.venv\Scripts\python app.py --selftest samples\INR18650P28A-V1-80093.pdf   # kütüphane/paket denetimi
```

Tek dosya `.exe`: `python build_exe.py` → `dist\PlotDigitizer.exe` (ilk açılış birkaç sn sürer).
`--screenshot out.png dosya.pdf --page gallery|detail --chart N --curve M` pencerenin görüntüsünü kaydeder.

## Yapı

```
app.py                     giriş noktası (+ --selftest / --screenshot / --manual)
core/
  vector_charts.py         PDF vektör verisinden grafik, eğri, anlam, eksen uyumu, notlar
  pdf_source.py            PDF sayfa/bölge render, grafik çerçevesi, başlık, eksen/tick bulma
  lookup.py                X→Y tablosu, interpolasyon, ondalık basamak, HTML raporu, CSV
  query.py                 X→Y ve Y→X değer sorgusu (interpolasyon, çoklu çözüm, belirsizlik)
  curve_edit.py            elle düzeltme: nokta ekle/sil/taşı, geri al/yinele (GUI'siz)
  calibration_store.py     resim grafiklerin kalibrasyonunu PDF içeriğine göre hatırlar (JSON)
  raster_charts.py         PDF'e resim olarak gömülü grafikler: bulma, tick tespiti, kalibrasyon, otomatik eğri okuma
  postprocess.py, calibration.py, models.py
  extraction.py, project.py, plotarea.py, export.py, imageio.py    eski görselden sayısallaştırma
ui/
  lookup_window.py         ana pencere: hoş geldin → galeri → detay (düzeltme araçları, resim grafik kalibrasyonu)
  query_panel.py           "Değer sorgula" kutusu
  calibration_dialog.py    resim grafikte tıklanan işaretlerin değerleri
  printing.py              tabloyu PDF'e çevirir / yazıcıya gönderir (QTextDocument + QPrinter)
  image_view.py, table_model.py, appicon.py
  main_window.py, pdf_dialog.py                                    eski elle araç
tests/                     birim + entegrasyon + arayüz; matplotlib ile doğru cevabı bilinen PDF'ler
samples/                   INR18650P28A-V1-80093.pdf (datasheet), discharge.png
```

Tüm koordinatlar PDF noktası (pt) cinsindendir, sayfanın sol-üst köşesinden ölçülür (y aşağı).
