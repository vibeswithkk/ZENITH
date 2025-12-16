## **Cetak Biru Proyek Zenith: Framework Unifikasi dan Optimisasi Cross-Platform untuk Machine Learning**

**Nama Penyusun:** Wahyu Ardiansyah
**Tanggal:** 16 Desember 2025
**Status Dokumen:** Cetak Biru Arsitektur Utama – Versi 1.0

---

### **Daftar Isi**

**ABSTRAK** ... 3

**BAB 1: PENDAHULUAN** ... 4
1.1. Latar Belakang dan Permasalahan dalam Ekosistem ML Modern ... 4
1.2. Visi dan Misi Proyek Zenith ... 5
1.3. Ruang Lingkup dan Batasan Proyek ... 6
1.4. Metodologi Penyusunan Cetak Biru ... 6

**BAB 2: LANDASAN TEORI DAN STUDI LITERATUR** ... 7
2.1. Analisis Framework ML Dominan: PyTorch, TensorFlow, dan JAX ... 7
2.2. Prinsip Sistem AI yang Model-Agnostik ... 10
2.3. Teori Optimisasi dalam Machine Learning ... 11
2.4. Arsitektur untuk Platform Multi-Akselerator ... 13

**BAB 3: KONSEP DAN ARSITEKTUR ZENITH** ... 15
3.1. Filosofi Inti: Unifikasi, Performa, dan Universalitas ... 15
3.2. Arsitektur Berlapis Zenith ... 16
3.3. Prinsip Model-Agnostik dan Abstraksi Perangkat Keras ... 18
3.4. Diagram Alir (Flowchart) Operasi Zenith ... 19
3.5. Diagram Use Case Pengguna Zenith ... 20
3.6. Diagram Aktivitas Proses Optimisasi ... 21

**BAB 4: SPESIFIKASI TEKNIS DAN MATEMATIKA** ... 22
4.1. Fondasi Matematika dan Jaminan Numerik ... 22
4.2. Rumusan Optimisasi Inti ... 23
4.3. Spesifikasi Bahasa Pemrograman dan Framework Pendukung ... 25
4.4. Diagram Kelas (Class Diagram) Inti ... 27

**BAB 5: STRATEGI IMPLEMENTASI DAN PENGUJIAN** ... 28
5.1. Kerangka Implementasi 6-Fase ... 28
5.2. Strategi Pengembangan Perangkat Lunak ... 30
5.3. Strategi Pengujian (Testing) Menyeluruh ... 31
5.4. Integrasi ke dalam CI/CD Pipeline ... 33

**BAB 6: INTEGRASI DENGAN EKOSISTEM YANG ADA** ... 34
6.1. Mekanisme Integrasi dengan PyTorch, TensorFlow, dan JAX ... 34
6.2. Peran ONNX sebagai Format Pertukaran Netral ... 35
6.3. Strategi Abstraksi untuk Berbagai Perangkat Keras (CPU, GPU, TPU, NPU, FPGA) ... 36
6.4. Kompatibilitas dengan Alat dan Platform Lain (Hugging Face, dll.) ... 37

**BAB 7: NILAI DAN DAMPAK YANG DIHARAPKAN** ... 38
7.1. Value Proposition bagi Berbagai Pemangku Kepentingan ... 38
7.2. Analisis Trade-off yang Dikelola secara Proaktif ... 39
7.3. Dampak terhadap Kecepatan Riset dan Efisiensi Produksi ... 40

**BAB 8: RENCANA JALAN (ROADMAP) DAN KESIMPULAN** ... 41
8.1. Roadmap Pengembangan Tahapan ... 41
8.2. Kesimpulan dan Peringatan Implementasi ... 42
8.3. Langkah Selanjutnya ... 43

**DAFTAR PUSTAKA** ... 44
**LAMPIRAN** ... 50

---

### **ABSTRAK**

**Zenith** adalah cetak biru untuk sebuah *framework* pembantu (*helper framework*) yang berambisi menjadi lapisan unifikasi dan optimasi lintas-platform (*cross-platform*) dalam ekosistem *Machine Learning* (ML) dan *Deep Learning* (DL). Tantangan utama dalam pengembangan AI modern adalah fragmentasi yang diakibatkan oleh beragamnya *framework* (seperti PyTorch, TensorFlow, JAX) dan perangkat keras akselerator (GPU NVIDIA, AMD, TPU, NPU, FPGA) yang masing-masing memiliki ekosistem, API, dan karakteristik optimasi tersendiri. Fragmentasi ini menghambat portabilitas model, meningkatkan biaya pemeliharaan, dan mempersulit peneliti serta insinyur untuk mencapai performa optimal di berbagai lingkungan deployment.

Visi Zenith adalah menjadi lapisan abstraksi yang model-agnostik dan hardware-agnostik, yang menyediakan antarmuka yang konsisten dan seragam untuk mengeksekusi serta mengoptimasi model dari berbagai *framework* sumber di atas beragam perangkat keras target. Dengan pendekatan ini, Zenith tidak bertujuan menggantikan *framework* yang ada, melainkan melengkapi mereka dengan kemampuan untuk "bicara" dan berkinerja optimal di lingkungan yang lebih luas. Nilai inti yang ditawarkan mencakup peningkatan kecepatan pelatihan dan inferensi melalui kompilasi dan *kernel fusion* yang canggih, pengurangan penggunaan memori, serta preservasi akurasi yang dapat diverifikasi secara matematis.

Cetak biru ini merinci landasan teoritis berbasis studi literatur dan praktik industri, arsitektur sistem berlapis, spesifikasi teknis termasuk fondasi matematika untuk optimasi dan stabilitas numerik, serta strategi implementasi bertahap yang mencakup siklus pengujian yang ketat. Zenith dirancang untuk mengatasi *trade-off* klasik antara akurasi, kecepatan, dan efisiensi sumber daya dengan memberikan kontrol yang lebih cerdas dan terukur kepada pengguna. Keberhasilannya diharapkan dapat mendemokratisasi akses ke komputasi berperforma tinggi, mempercepat siklus riset, dan menyederhanakan alur kerja produksi AI dari lab hingga deployment di edge, cloud, atau perangkat khusus.

---

### **BAB 1: PENDAHULUAN**

#### **1.1. Latar Belakang dan Permasalahan dalam Ekosistem ML Modern**
Dunia ML saat ini didominasi oleh beberapa *framework* besar dengan filosofi dan kekuatan berbeda. **PyTorch**, dengan *execution graph* yang dinamis dan *Pythonic*, telah menjadi favorit komunitas riset (digunakan oleh lebih dari 75% makalah DL baru). **TensorFlow**, dengan ekosistem produksi yang matang (seperti TensorFlow Serving, Lite, dan.js) serta dukungan kuat untuk TPU melalui compiler XLA, tetap menjadi pilihan utama untuk sistem berskala produksi. **JAX**, yang menawarkan komposisi fungsi murni, diferensiasi otomatis, dan kompilasi JIT (*Just-In-Time*) melalui XLA, memberikan performa dan skalabilitas yang luar biasa, terutama untuk komputasi ilmiah dan model berskala sangat besar.

Namun, keberagaman ini menciptakan beberapa masalah signifikan:
1.  **Hambatan Portabilitas Model**: Model yang dilatih di satu *framework* sulit untuk dieksekusi atau dioptimasi secara optimal di *framework* lain atau pada perangkat keras yang berbeda tanpa usaha konversi yang signifikan.
2.  **Duplikasi Upaya Optimasi**: Setiap vendor perangkat keras dan *framework* harus mengembangkan dan memelihara *kernel* komputasi yang dioptimasi sendiri-sendiri untuk operasi dasar (seperti *matrix multiplication* dan konvolusi), yang merupakan pekerjaan duplikatif dan mahal.
3.  **Kompleksitas Manajemen Sumber Daya**: Pengguna yang ingin memanfaatkan heterogenitas perangkat keras (misalnya, beberapa GPU dari vendor berbeda, atau CPU dengan instruksi vektor spesifik) harus menulis kode pengelolaan yang kompleks dan spesifik.
4.  **Trade-off yang Tidak Terkelola dengan Baik**: Pengembang sering kali dihadapkan pada pilihan sulit antara akurasi, latensi, dan konsumsi daya, tanpa alat yang memadai untuk mengeksplorasi dan mengelola *trade-off* ini secara sistematis.

#### **1.2. Visi dan Misi Proyek Zenith**
Visi Zenith adalah menjadi lapisan perangkat lunak yang menjadi *de facto standard* untuk eksekusi dan optimasi model ML yang portabel, berkinerja tinggi, dan efisien sumber daya, di atas *framework* dan perangkat keras apa pun.

Misi untuk mewujudkan visi ini adalah:
1.  **Menyediakan Antarmuka yang Model-Agnostik**: Mengizinkan model dari PyTorch, TensorFlow, JAX, dan lainnya untuk dijalankan melalui API Zenith yang seragam tanpa modifikasi kode sumber yang signifikan.
2.  **Mengabstraksi Kompleksitas Perangkat Keras**: Memberikan satu antarmuka pemrograman untuk menargetkan berbagai akselerator (CPU, GPU, TPU, NPU, FPGA), dengan *runtime* yang secara otomatis memilih *kernel* yang paling optimal.
3.  **Menerapkan Optimasi yang Secara Matematis Terjamin**: Mengintegrasikan teknik kompilasi grafik, *kernel fusion*, kuantisasi, dan *pruning* yang tidak hanya heuristik tetapi didukung oleh prinsip optimasi matematis untuk menjamin kecepatan tanpa mengorbankan akurasi yang telah ditentukan.
4.  **Menyederhanakan Manajemen Siklus Hidup Model**: Menjadi penghubung yang mulus antara fase riset (di PyTorch) dan fase produksi (di TensorFlow Serving atau lingkungan edge), mengurangi *friction* dalam alur kerja MLOps.

#### **1.3. Ruang Lingkup dan Batasan Proyek**
Cetak biru ini membatasi ruang lingkup pada:
*   **Fase Inferensi dan Pelatihan**: Zenith akan mengoptimasi kedua fase, dengan prioritas awal pada inferensi yang lebih kompleks dalam hal deployment.
*   **Framework Sumber Awal**: Dukungan pertama akan difokuskan pada PyTorch, TensorFlow, dan JAX/Flax sebagai *frontend* utama.
*   **Perangkat Keras Target Awal**: CPU (x86/ARM), GPU NVIDIA (CUDA), GPU AMD (ROCm), dan Google TPU akan menjadi target dukungan pertama.
*   **Model Zenith sebagai *Helper***: Zenith secara eksplisit dirancang sebagai pelengkap, bukan pengganti. Ia akan diintegrasikan sebagai *library* atau *toolchain* yang dipanggil oleh *framework* yang ada.

Batasan termasuk tidak mengembangkan bahasa pemrograman baru, tidak menggantikan compiler tingkat rendah seperti LLVM sepenuhnya, dan mengandalkan *driver* perangkat keras dari vendor.

#### **1.4. Metodologi Penyusunan Cetak Biru**
Cetak biru ini disusun berdasarkan:
1.  **Studi Literatur dan Analisis Komparatif**: Menganalisis kekuatan, kelemahan, dan arsitektur *framework* yang ada serta prinsip desain sistem yang *model-agnostic*.
2.  **Analisis Studi Kasus Industri**: Memelajari implementasi nyata sistem multi-akselerator dan *framework* unifikasi.
3.  **Sintesis Teori Optimisasi**: Merujuk pada teori dan algoritma optimisasi ML yang mapan untuk membangun fondasi matematika yang kuat.
4.  **Penerapan Praktik Rekayasa Perangkat Lunak Terbaik**: Mengadopsi metodologi pengembangan bertahap dan standar pengujian yang ketat dalam rencana implementasi.

---

### **BAB 2: LANDASAN TEORI DAN STUDI LITERATUR**

#### **2.1. Analisis Framework ML Dominan: PyTorch, TensorFlow, dan JAX**
Pemahaman mendalam tentang *framework* yang ada sangat penting untuk merancang lapisan kompatibilitas Zenith. Berikut adalah analisis komparatif berdasarkan penelitian yang ada:

*   **PyTorch**: Keunggulan utama terletak pada **execution graph yang dinamis** (*eager execution*), yang selaras dengan alur pemrograman Python dan memudahkan proses *debugging* serta eksperimen yang cepat. Filosofi "Pythonic" ini telah membuatnya dominan di lingkup akademik dan riset. API-nya yang intuitif untuk diferensiasi otomatis (*autograd*) dan manipulasi tensor mendukung prototipe yang gesit. Namun, untuk deployment produksi berskala besar, grafik dinamis dapat menimbulkan overhead. Proyek seperti TorchScript dan kompiler `torch.compile` (dengan TorchDynamo dan Inductor) adalah upaya internal PyTorch untuk mengatasi ini dengan mengonversi kode ke grafik statis yang dapat dioptimasi, sebuah ruang di mana Zenith dapat beroperasi dengan lebih agnostik.

*   **TensorFlow**: Awalnya dibangun di sekitar **execution graph yang statis**, TensorFlow 2.x mengadopsi *eager execution* sebagai default, namun tetap mempertahankan kekuatan ekosistem produksinya yang komprehensif. Kekuatan utamanya adalah **TFX** (*TensorFlow Extended*) untuk alur kerja MLOps, **TensorFlow Serving** untuk deployment model, **TensorFlow Lite** untuk perangkat edge, dan **TensorFlow.js** untuk web. Dukungannya yang kuat dan native untuk compiler **XLA** (*Accelerated Linear Algebra*) dan **TPU** Google menjadikannya sangat powerful untuk pelatihan dan inferensi skala sangat besar. *Trade-off*-nya adalah, dibandingkan PyTorch, ekosistemnya dianggap kurang fleksibel untuk riset eksploratif yang cepat.

*   **JAX**: JAX mengambil pendekatan fundamental yang berbeda. Ia adalah *framework* untuk **transformasi komposisi fungsi** (seperti `grad`, `jit`, `vmap`, `pmap`) pada kode yang mirip NumPy. Filosofi **fungsi murni** (*pure functions*) ini memungkinkan determinisme dan memudahkan kompilasi serta paralelisasi. Kompilasi **JIT** melalui **XLA** adalah fitur andalannya, yang sering menghasilkan percepatan kinerja yang signifikan. JAX beroperasi pada tingkat yang lebih rendah dibandingkan PyTorch atau Keras, sehingga *library* seperti **Flax** atau **Haiku** dibutuhkan untuk membangun jaringan saraf dengan mudah. Kekuatannya terletak pada skalabilitas dan performa tinggi untuk komputasi ilmiah dan model besar, terutama di lingkungan TPU.

**Tabel 2.1: Analisis Komparatif Framework ML**

| Aspek | **PyTorch** | **TensorFlow (+Keras)** | **JAX (+Flax)** | **Implikasi untuk Zenith** |
| :--- | :--- | :--- | :--- | :--- |
| **Paradigma Eksekusi** | Dinamis (Eager) pertama, Grafik Statis (via kompilasi) | Gabungan (Eager default, Grafik via `tf.function`)| Fungsi Murni, dikompilasi JIT via XLA | Zenith harus menangani kedua mode: tracing grafik dari kode eager dan mengonsumsi fungsi JAX. |
| **Komunitas & Ekosistem** | Sangat kuat di riset, dukungan luas dari Hugging Face. | Sangat kuat di produksi, alat deployment matang. | Berkembang pesat, terutama di kalangan peneliti Google/DeepMind. | Harus mendukung model dari ekosistem yang berbeda (e.g., `.pt`, SavedModel, Flax params). |
| **Optimasi & Kompilasi** | `torch.compile` (Inductor), mendukung XLA/TPU. | XLA adalah inti, optimasi grafik otomatis. | XLA adalah inti, kompilasi JIT agresif. | Dapat memanfaatkan XLA sebagai *backend* kompilasi umum, atau menyediakan *compiler* alternatif. |
| **Kekuatan Utama** | Fleksibilitas, *debugging* mudah, prototipe cepat. | Stabilitas produksi, skalabilitas, ekosistem lengkap. | Performa, determinisme, skalabilitas ekstrem (TPU). | Zenith harus mempertahankan kekuatan masing-masing saat berinteraksi dengannya. |
| **Kelemahan untuk Unifikasi** | Grafik dinamis menambah kompleksitas untuk optimasi lintas platform. | API yang berkembang dan kompleksitas ekosistem. | Kurva belajar lebih curam, model dalam fungsi murni. | Membutuhkan *adapter* yang cerdas untuk setiap paradigma. |

#### **2.2. Prinsip Sistem AI yang Model-Agnostik**
Untuk menjadi lapisan yang efektif, Zenith harus dibangun berdasarkan prinsip-prinsip **model-agnostik**. Menurut , ini berarti memisahkan logika aplikasi atau sistem dari model AI spesifik yang menjalankan tugas. Prinsip-prinsip kunci yang diadopsi untuk Zenith adalah:

1.  **Dekoupling Logika dari Inferensi**: Sistem Zenith harus mendefinisikan **"apa"** (tugas: klasifikasi, generasi, dll.) terpisah dari **"bagaimana"** (model PyTorch mana, di GPU mana). Abstraksi ini memungkinkan pertukaran model tanpa mengubah kode sistem induk.
2.  **Abstraksi API yang Konsisten**: Mengadopsi atau terinspirasi oleh standar API yang muncul (seperti OpenAI-compatible API) dapat memastikan portabilitas dan mempermudah integrasi dengan berbagai *backend* model.
3.  **Modularitas**: Setiap komponen dalam Zenith, seperti *converter* model, *optimizer*, atau *hardware runtime*, harus dirancang sebagai modul yang dapat dipasang dan ditingkatkan secara independen. Ini memungkinkan dukungan untuk *framework* atau perangkat keras baru ditambahkan tanpa mengganggu inti sistem.
4.  **Observabilitas dan Evaluasi Berkelanjutan**: Sistem harus mampu mengukur kinerja model (latensi, akurasi, penggunaan memori) dalam konteks nyata. Data ini penting tidak hanya untuk *monitoring*, tetapi juga untuk membuat keputusan cerdas secara *runtime*, seperti memilih *backend* yang optimal untuk kondisi beban kerja saat ini.

#### **2.3. Teori Optimisasi dalam Machine Learning**
Fondasi matematika Zenith berasal dari teori optimisasi, yang terbagi menjadi dua kategori utama yang relevan: **Optimisasi Model** (*Optimization I*) dan **Optimisasi dengan Model** (*Optimization II*). Zenith terutama bergerak di ranah *Optimization I*, yaitu meningkatkan kinerja model ML itu sendiri.

*   **Algoritma Optimisasi Dasar**: Inti pelatihan model sering kali adalah **Gradient Descent** dan variannya. Aturan pembaruan parameter \( \theta \) pada iterasi \( t \) adalah:
    \[
    \theta_{t+1} = \theta_t - \eta \cdot \nabla J(\theta_t)
    \]
    di mana \( \eta \) adalah **learning rate** dan \( \nabla J(\theta_t) \) adalah gradien fungsi kerugian (*loss*). Zenith dapat mengoptimasi penerapan operasi ini, misalnya dengan memilih implementasi matriks yang paling efisien untuk kalkulasi gradien.

*   **Optimizer Modern**: Algoritma seperti **Adam** (Adaptive Moment Estimation), yang menggabungkan *momentum* dan penyesuaian *learning rate* per-parameter, adalah standar de facto. Zenith dapat mengintegrasikan kernel yang sangat dioptimalkan untuk *optimizer* ini di berbagai perangkat keras.

*   **Fungsi Kerugian dan Lanskap Kerugian (*Loss Landscape*)**: Proses pelatihan melibatkan minimisasi fungsi kerugian \( J(\theta) \). "Medan" dari fungsi ini terhadap parameter disebut *loss landscape*. Teknik seperti *mixed precision training* (menggunakan FP16/FP32 campuran) yang dapat diakselerasikan oleh Zenith, memengaruhi navigasi di lanskap ini dan perlu dikelola untuk menjaga stabilitas.

*   **Optimisasi Inferensi**: Ini adalah fokus utama Zenith. Tekniknya mencakup:
    *   **Kuantisasi**: Mengurangi presisi numerik bobot dan aktivasi (dari FP32 ke INT8/INT4) untuk mengurangi ukuran model dan meningkatkan kecepatan inferensi.
    *   **Pruning**: Membuang koneksi atau neuron yang kurang penting dalam model untuk menciptakan model yang renggang (*sparse*) dan lebih efisien.
    *   ***Kernel Fusion***: Menggabungkan beberapa operasi berturut-turut (misalnya, konvolusi, normalisasi batch, dan aktivasi ReLU) menjadi satu operasi *kernel* tunggal untuk mengurangi overhead memori dan loncatan antar *kernel*.
    *   ***Graph Optimization***: Menganalisis dan menulis ulang grafik komputasi model untuk menghilangkan operasi redundan, menyusun ulang operasi untuk lokalisasi cache yang lebih baik, dll.

    **Teorema dan Jaminan**: Bagian dari riset Zenith melibatkan formulasi jaminan matematis. Sebagai contoh, untuk kuantisasi, kita ingin memastikan bahwa untuk input \( x \), kesalahan antara output model asli \( f(x) \) dan model terkuantisasi \( \hat{f}(x) \) dibatasi:
    \[
    \| f(x) - \hat{f}(x) \| \leq \epsilon \cdot \|x\| + \delta
    \]
    di mana \( \epsilon \) dan \( \delta \) adalah konstanta kecil yang diturunkan dari teknik kuantisasi tertentu. Zenith bertujuan untuk menyediakan alat yang secara otomatis memverifikasi atau menegaskan batasan semacam ini selama proses optimasi.

#### **2.4. Arsitektur untuk Platform Multi-Akselerator**
Studi kasus dari MulticoreWare memberikan contoh konkret tentang tantangan dan solusi dalam membangun *framework* AI yang berjalan di atas perangkat keras yang heterogen (NPU, GPU dari vendor berbeda). Beberapa wawasan kunci yang diadopsi oleh Zenith adalah:

*   ***Abstraction of Accelerator Diversity***: Lapisan inti Zenith harus menyembunyikan kompleksitas *driver*, *runtime*, dan memori spesifik dari setiap akselerator. Aplikasi pengguna hanya berinteraksi dengan API Zenith yang seragam.
*   ***Unified Installer and Environment Setup***: Menyediakan skrip atau alat yang menyederhanakan penyiapan lingkungan perangkat lunak yang diperlukan (seperti *driver* CUDA, ROCm, *runtime* OpenCL) di berbagai platform.
*   ***API Kompatibel dan Runtime Switching***: Mengimplementasikan API (misalnya, kompatibel dengan OpenAI) yang memungkinkan pengguna atau *orchestrator* untuk secara dinamis memilih akselerator target (misalnya, GPU vs. NPU) berdasarkan ketersediaan, beban, atau kebutuhan daya.
*   ***Model Integration and Optimization***: Mendukung integrasi model dari berbagai sumber (seperti Qwen, Llama, model difusi) dan menerapkan optimasi tingkat sistem seperti pengelolaan ***KV Cache*** untuk transformer, paralelisme model/data, dan format presisi campuran (seperti W4A16, FP16) untuk meningkatkan throughput inferensi.

Dengan mempelajari pendekatan ini, Zenith dapat dirancang bukan sebagai *framework* yang terisolasi, tetapi sebagai komponen dalam tumpukan perangkat lunak yang lebih luas yang memungkinkan AI berjalan di mana saja.

---

### **BAB 3: KONSEP DAN ARSITEKTUR ZENITH**

Bab ini akan menguraikan desain fundamental dan struktur sistem Zenith. Kami akan memulai dengan filosofi inti yang mendorong setiap keputusan arsitektural, kemudian menjelaskan arsitektur berlapis yang mewujudkannya, dan diakhiri dengan prinsip-prinsip operasional serta diagram yang menggambarkan alur kerja sistem.

#### **3.1. Filosofi Inti: Unifikasi, Performa, dan Universalitas**
Zenith didirikan di atas tiga pilar filosofis yang tidak dapat dinegosiasikan. Ketiganya saling terkait dan memandu setiap aspek desain.

1.  **Unifikasi tanpa Invasif (The Glue, Not The Replacement)**: Zenith secara eksplisit dirancang sebagai *framework* pembantu. Filosofi ini diilhami oleh keberhasilan alat seperti **ONNX Runtime** dan **Apache TVM**, yang berfungsi sebagai lapisan eksekusi netral di atas berbagai *framework*. Zenith tidak bermaksud menggantikan PyTorch, TensorFlow, atau JAX, tetapi menjadi "lem" yang memungkinkan mereka bekerja bersama dan mencapai potensi optimal di lingkungan yang lebih beragam. Tujuannya adalah **zero switching cost** bagi pengguna—kode yang ada dapat ditingkatkan dengan satu baris impor `import zenith`, tanpa perlu menulis ulang logika.

2.  **Performa dengan Jaminan (Guaranteed Speed, Preserved Accuracy)**: Komitmen Zenith adalah memberikan peningkatan performa yang terukur dan dapat diandalkan. Ini melampaui optimasi heuristik; ia bertujuan untuk optimasi yang didukung oleh **fondasi matematika yang ketat** (seperti yang akan dijelaskan di Bab 4). Setiap transformasi (kuantisasi, *pruning*, *fusion*) harus dilengkapi dengan batasan kesalahan (*error bounds*) yang dapat diverifikasi. Pengguna dapat menentukan target (misalnya, "percepat 3x dengan akurasi tidak turun lebih dari 0.1%"), dan Zenith akan mencari konfigurasi optimasi yang memenuhi spesifikasi tersebut atau gagal dengan jelas jika tidak mungkin.

3.  **Universalitas melalui Abstraksi (Write Once, Run Anywhere on Anything)**: Zenith bercita-cita menjadi benar-benar *hardware-agnostic*. Prinsip ini, yang juga menjadi inti dari pendekatan **oneAPI** Intel dan visi **Vulkan** untuk grafik, diterapkan ke domain ML. Melalui lapisan abstraksi perangkat keras (*Hardware Abstraction Layer - HAL*), model yang dikembangkan di satu *framework* harus dapat dieksekusi secara optimal di CPU, GPU NVIDIA, GPU AMD, TPU, NPU, atau FPGA tanpa perubahan kode. Abstraksi ini juga mencakup manajemen memori yang canggih (mis., dukungan **NUMA** untuk sistem multi-socket, **Unified Memory**).

#### **3.2. Arsitektur Berlapis Zenith**
Untuk mencapai filosofi di atas, Zenith mengadopsi arsitektur berlapis (*layered architecture*) yang modular. Setiap lapisan memiliki tanggung jawab yang jelas dan berkomunikasi melalui antarmuka yang terdefinisi dengan baik. Pendekatan ini memfasilitasi pengembangan, pengujian, dan pemeliharaan.

**Gambaran Arsitektur Tingkat Tinggi:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Python User Interface                    │
│           `import zenith; zenith.optimize(model)`           │
├─────────────────────────────────────────────────────────────┤
│              Framework-Specific Adapters Layer              │
│      (PyTorch, TensorFlow, JAX, ONNX, etc. Importers)       │
├─────────────────────────────────────────────────────────────┤
│       Core Optimization & Compilation Engine (C++/Rust)     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  High-Level Graph Optimizer & IR (MLIR/Graph-level)  │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │   Kernel Scheduler & Auto-Tuner (Hardware-aware)     │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │       Mathematical Kernel Library (Zenith-MKL)       │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│           Hardware Abstraction Layer (HAL)                  │
│   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐        │
│   │ CUDA │ROCM  │ SYCL │ Metal│Vulkan│ TPU  │ CPU  │ ...    │
│   │Runtime│Runtime│Runtime│Runtime│Runtime│Runtime│Backend│ │
│   └──────┴──────┴──────┴──────┴──────┴──────┴──────┘        │
├─────────────────────────────────────────────────────────────┤
│                Physical Hardware Resources                  │
│      (NVIDIA GPU, AMD GPU, Intel CPU/GPU, TPU, etc.)        │
└─────────────────────────────────────────────────────────────┘
```

**Penjelasan Lapisan:**

1.  **Python User Interface (Thin Wrapper)**: Lapisan ini menyediakan API Python yang sederhana dan intuitif. Ia dirancang minimalis untuk meminimalkan overhead dan mempermudah adopsi. Fungsinya terutama untuk menerima model atau kode pengguna, meneruskan ke lapisan adapter, dan mengembalikan model yang dioptimalkan. Contoh: `optimized_model = zenith.compile(model, target='cuda', precision='fp16')`.

2.  **Framework-Specific Adapters Layer**: Lapisan ini bertanggung jawab untuk "memahami" berbagai *framework* sumber. Setiap *adapter* (mis., `PyTorchAdapter`, `TFAdapter`, `JAXAdapter`, `ONNXAdapter`) akan:
    *   **Melacak/Mengekspor Grafik Komputasi**: Untuk PyTorch, ini mungkin menggunakan `torch.export` atau `torch.jit.trace`. Untuk TensorFlow, menggunakan `tf.function` dan Grafik. Untuk JAX, menerima fungsi yang sudah dikompilasi JIT.
    *   **Mengonversi ke Perantara Zenith (Intermediate Representation - IR)**: Grafik dari berbagai sumber dikonversi ke format IR internal Zenith yang seragam. **ONNX** berperan penting di sini sebagai format pertukaran yang netral; banyak *adapter* dapat mengonversi ke ONNX terlebih dahulu sebelum diproses lebih lanjut oleh Zenith.

3.  **Core Optimization & Compilation Engine**: Ini adalah "otak" Zenith, ditulis dalam C++/Rust untuk performa maksimal. Ia terdiri dari beberapa sub-modul:
    *   **High-Level Graph Optimizer**: Menganalisis dan mengubah IR menggunakan teknik seperti *dead code elimination*, *constant folding*, *operator fusion*, dan *layout transformation*. Kami mempertimbangkan penggunaan **MLIR** (Multi-Level Intermediate Representation) sebagai basis IR karena kemampuannya untuk merepresentasikan komputasi dari level tinggi hingga rendah dalam satu *framework*.
    *   **Kernel Scheduler & Auto-Tuner**: Modul ini membuat jadwal eksekusi yang optimal dengan mempertimbangkan hierarki memori, dependensi data, dan ketersediaan unit komputasi. Ini juga dapat melakukan *auto-tuning*, yaitu menjalankan berbagai varian *kernel* untuk satu operasi pada *hardware* target untuk memilih yang tercepat.
    *   **Mathematical Kernel Library (Zenith-MKL)**: Seperangkat implementasi operasi matematika (matmul, konvolusi, normalisasi, dll.) yang sangat dioptimasi untuk berbagai arsitektur. Ini akan mengintegrasikan dan memperluas *library* yang ada seperti **oneDNN** (Intel), **cuDNN** (NVIDIA), dan **rocBLAS** (AMD), dengan wrapper seragam.

4.  **Hardware Abstraction Layer (HAL)**: Lapisan ini mengabstraksi detail spesifik vendor dari *driver* dan *runtime* perangkat keras. Setiap *backend* (CUDA, ROCm, SYCL, dll.) menerjemahkan perintah dan *kernel* generik dari lapisan inti menjadi panggilan API native. HAL juga menangani alokasi dan transfer memori yang efisien di seluruh perangkat.

5.  **Physical Hardware**: Lapisan paling bawah adalah perangkat keras fisik itu sendiri. Zenith bertujuan untuk mendukung spektrum yang seluas mungkin.

#### **3.3. Prinsip Model-Agnostik dan Abstraksi Perangkat Keras**
Dua prinsip teknis utama yang memungkinkan filosofi Zenith adalah:

*   **Model-Agnostik melalui Representasi Menengah (IR)**: Kunci dari unifikasi adalah mengonversi model dari semua *framework* ke dalam **Representasi Menengah (IR)** Zenith yang umum. IR ini harus mampu menangkap semantics komputasi dari model apa pun (operator, tensor, aliran data). **ONNX** adalah contoh praktis dari prinsip ini; ia mendefinisikan sekumpulan operator standar. Zenith dapat menggunakan ONNX sebagai IR awal atau mendefinisikan IR-nya sendiri yang lebih kaya, mungkin berdasarkan **MLIR**, yang dapat mengekspresikan paralelisme, tipe data kuantisasi, dan konstruksi tingkat rendah secara lebih baik.

*   **Hardware-Agnostic melalui Kernel Dispatch dan Seleksi Otomatis**: Lapisan HAL memungkinkan penulisan logika optimasi sekali untuk banyak *backend*. Mekanisme *kernel dispatch* bekerja seperti ini: Saat Core Engine perlu mengeksekusi suatu operasi (misalnya, `Conv2D`), ia memanggil antarmuka umum HAL. Berdasarkan perangkat target yang aktif, HAL memuat implementasi *kernel* yang tepat. Lebih canggih lagi, **Auto-Tuner** dapat memiliki beberapa *kernel* untuk satu operasi yang sama (misalnya, satu menggunakan *winograd algorithm*, satu lagi *direct convolution*). Pada saat inisialisasi atau *runtime*, ia dapat menjalankan *benchmark* mikro untuk memilih *kernel* terbaik untuk ukuran dan bentuk input spesifik pada *hardware* saat itu.

#### **3.4. Diagram Alir (Flowchart) Operasi Zenith**
Diagram berikut menggambarkan alur kerja tipikal saat pengguna memanggil Zenith untuk mengoptimasi model.

```
┌─────────────────┐
│   Model Input   │ (PyTorch nn.Module, TF SavedModel,
│   dari Pengguna │  JAX function, atau file ONNX)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Framework-Spec  │
│    Adapter      │ → Ekspor/konversi ke IR Zenith
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Graph Analysis │ → Analisis statis: dependensi,
│   & Optimisasi  │   bottleneck, peluang fusion.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Target-Specific │ → Berdasarkan target (cuda, cpu),
│   Compilation   │   pilih kernel & generate code.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │ → Hasil: Executable, library,
│     Output      │   atau model runtime yang siap.
└─────────────────┘
```

**Keterangan Alur:**
1.  **Input**: Pengguna memberikan model yang sudah ada.
2.  **Adaptasi**: *Adapter* yang sesuai mengekstrak grafik komputasi.
3.  **Analisis & Optimisasi**: Engine inti menganalisis grafik dan menerapkan serangkaian pass optimisasi.
4.  **Kompilasi**: Grafik yang dioptimasi dikompilasi untuk target perangkat keras tertentu, memilih *kernel* yang optimal.
5.  **Output**: Menghasilkan artefak yang dioptimasi yang dapat dijalankan di lingkungan target.

#### **3.5. Diagram Use Case Pengguna Zenith**
Diagram *use case* UML berikut mengidentifikasi aktor utama dan bagaimana mereka berinteraksi dengan sistem Zenith.

```
                      ┌─────────────────────┐
                      │     Sistem Zenith   │
                      └─────────────────────┘
                               △ △ △
               ┌───────────────┼ │ └───────────────┐
               │               │ │                 │
    ┌──────────┴────────┐ ┌───┴─┴──────┐ ┌────────┴──────────┐
    │   AI Researcher   │ │ ML Engineer │ │   System Admin   │
    └──────────┬────────┘ └───┬─┬──────┘ └────────┬──────────┘
               │              │ │                 │
         ┌─────┴─────┐ ┌─────┴─┴─────┐     ┌─────┴─────┐
         │ Gunakan   │ │ Integrasikan│     │ Deploy &  │
         │ di Riset  │ │ ke Pipeline │     │  Monitor  │
         └───────────┘ │  Produksi   │     └───────────┘
                       └──────────────┘
```

**Aktor dan Tujuan:**
*   **AI Researcher**:
    *   *Use Case*: "Gunakan di Riset". Menggunakan Zenith untuk mempercepat eksperimen pelatihan model di workstation mereka (mungkin dengan beberapa GPU), atau untuk mengekspor model dari PyTorch ke format yang lebih efisien untuk demo.
*   **ML Engineer**:
    *   *Use Case*: "Integrasikan ke Pipeline Produksi". Menyematkan panggilan Zenith ke dalam alur kerja CI/CD MLOps (mis., sebagai tahap pasca-pelatihan) untuk secara otomatis mengoptimasi dan mempersiapkan model untuk deployment di server atau cloud.
*   **System Admin/DevOps**:
    *   *Use Case*: "Deploy & Monitor". Men-deploy *runtime* Zenith di cluster (mis., sebagai bagian dari *inference server*), mengonfigurasi alokasi sumber daya, dan memantau performa serta kesehatan sistem.

#### **3.6. Diagram Aktivitas Proses Optimisasi**
Diagram aktivitas UML ini merinci proses internal yang terjadi dalam *Core Optimization Engine*.

```
┌─────────────────────────────────────────────────────────────┐
│                  Aktivitas: Optimize Model                  │
└───────────────────────────────┬─────────────────────────────┘
                                ▼
                 ┌────────────────────────────┐
                 │   Terima IR dari Adapter   │
                 └──────────────┬─────────────┘
                                ▼
                 ┌────────────────────────────┐
                 │   Analisis Grafik (Static) │
                 │  - Identifikasi Subgraph   │
                 │  - Hitung FLOPs/Memori     │
                 └──────────────┬─────────────┘
                                ▼
      ┌──────────────────────────────────────────┐
      │     Loop untuk setiap Teknik Optimisasi  │
      └───────────────────┬──────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│ Evaluasi Pattern │           │  Terapkan jika   │
│   untuk Fusion   │           │  Memenuhi Syarat │
│   (Conv+ReLU,    │           │  (Percepatan >   │
│    MatMul+Add)   │           │   Threshold)     │
└────────┬─────────┘           └────────┬─────────┘
         │                               │
         └──────────────┬────────────────┘
                        ▼
         ┌────────────────────────────┐
         │  Update Grafik yang telah  │
         │       dioptimasi           │
         └──────────────┬─────────────┘
                        ▼
         ┌────────────────────────────┐
         │  Pilih Kernel & Generate   │
         │       Code Final           │
         └────────────────────────────┘
```

**Penjelasan Aktivitas:**
1.  **Terima IR**: Engine menerima representasi grafik seragam.
2.  **Analisis Statis**: Melakukan analisis awal untuk memahami struktur model dan mengidentifikasi area yang berpotensi untuk dioptimasi.
3.  **Loop Optimisasi**: Engine secara berurutan atau iteratif menerapkan berbagai *pass* optimisasi (fusion, layout transformation, dll.). Setiap *pass* memiliki kondisi (*pattern matching*) dan metrik (apakah transformasi ini menguntungkan untuk *hardware* target?).
4.  **Generate Code Akhir**: Setelah semua transformasi diterapkan, grafik yang telah dioptimasi diturunkan (*lowered*) menjadi kode atau *kernel* yang dapat dieksekusi untuk *backend* target tertentu.

Bab ini telah menjelaskan "mengapa" dan "bagaimana" konseptual dari Zenith. Bab selanjutnya akan mendalami "apa" secara teknis: rumus matematika, bahasa pemrograman, dan detail implementasi yang membentuk fondasi sistem ini.

---
### **BAB 4: SPESIFIKASI TEKNIS DAN MATEMATIKA**

Bab ini merinci fondasi teknis yang memungkinkan Zenith beroperasi sesuai dengan filosofinya. Kami akan membahas jaminan numerik dan rumus matematika inti, menentukan tumpukan teknologi (bahasa dan *framework*), serta memberikan gambaran desain kelas utama dalam sistem.

#### **4.1. Fondasi Matematika dan Jaminan Numerik**
Operasi Zenith harus deterministik dan dapat diprediksi. Untuk itu, kami mendefinisikan jaminan numerik sebagai bagian dari kontrak API.

**1. Jaminan Stabilitas Numerik (Numerical Stability Guarantee):**
Setiap transformasi optimasi `T` yang diterapkan Zenith pada suatu fungsi `F(x)` (misalnya, sebuah lapisan jaringan) harus mematuhi batasan kesalahan relatif yang dapat dikonfigurasi:
\[
\text{Untuk semua } x \in \text{Domain}, \quad \frac{| T(F)(x) - F(x) |}{|F(x)| + \epsilon} \leq \delta
\]
Di mana:
- `T(F)(x)` adalah output fungsi yang telah dioptimasi.
- `F(x)` adalah output fungsi asli (referensi).
- `ε` (epsilon) adalah konstanta kecil untuk menghindari pembagian dengan nol.
- `δ` (delta) adalah **batas toleransi kesalahan relatif maksimum**. Nilai default yang diusulkan adalah `δ = 1e-6` untuk presisi tunggal (FP32) dan `δ = 1e-3` untuk presisi setengah (FP16) dalam konteks inferensi. Nilai ini dapat disesuaikan oleh pengguna (`zenith.optimize(..., tolerance=1e-4)`).

**2. Jaminan Ketepatan Gradien (Gradient Fidelity Guarantee):**
Untuk optimasi yang memengaruhi fase pelatihan (seperti *kernel fusion* tingkat gradien), kami menjamin kesalahan gradien terbatas. Untuk parameter model `θ` dan fungsi kerugian `L`, gradien yang dihitung setelah optimasi, `∇̂L(θ)`, harus mendekati gradien asli `∇L(θ)`:
\[
\| \nabla \hat{L}(\theta) - \nabla L(\theta) \|_2 \leq \gamma \cdot \| \nabla L(\theta) \|_2
\]
Di mana `γ` (gamma) adalah koefisien yang mengontrol deviasi maksimum (misalnya, `γ = 0.001`). Ini memastikan bahwa jalur optimasi tidak menyimpang secara signifikan.

**3. Presisi Campuran yang Terkelola (Managed Mixed Precision):**
Zenith akan menerapkan presisi campuran (FP16/BF16 dengan FP32) secara aman. Alih-alih konversi buta, ia menggunakan analisis rentang dinamis (*dynamic range analysis*) untuk mengidentifikasi tensor yang rentan terhadap *overflow/underflow*. Untuk suatu operasi `op` dengan input `x`, presisi `P` (FP16) hanya akan digunakan jika:
\[
\max(|x|) < \text{MaxBound}(P) \cdot \alpha \quad \text{dan} \quad \min(|x|) > \text{MinBound}(P) / \alpha
\]
`α` (alpha, misalnya `0.95`) adalah faktor *safety margin*. Jika tidak terpenuhi, tensor tersebut akan tetap dalam FP32.

#### **4.2. Rumusan Optimisasi Inti**
Zenith menerjemahkan prinsip optimisasi menjadi rumus dan algoritma yang dapat diimplementasikan.

**1. Optimalisasi Penjadwalan Kernel dan Fusi:**
Tujuan *kernel fusion* adalah meminimalkan total waktu eksekusi dan transfer memori. Untuk serangkaian `n` operasi berurutan `{op₁, op₂, ..., opₙ}`, Zenith mengevaluasi biaya eksekusi terpisah vs. biaya eksekusi tergabung.
- **Biaya Terpisah**: `Cost_separate = Σ (T_exec(op_i) + T_mem(op_i))`, di mana `T_mem` adalah overhead baca/tulis tensor antara.
- **Biaya Fusi**: `Cost_fused = T_exec(fused_op) + T_mem(input) + T_mem(output)`.
Fusi akan diterapkan hanya jika `Cost_fused < Cost_separate * β`, dengan `β` (beta, misalnya `0.8`) adalah faktor ambang batas untuk memastikan peningkatan yang cukup signifikan. Pencarian pola fusi (seperti `Conv → BatchNorm → ReLU`) dilakukan menggunakan *graph pattern matching*.

**2. Optimasi Alokasi Memori dengan Checkpointing:**
Untuk pelatihan model besar, Zenith mengimplementasikan *gradient checkpointing* (atau *rematerialization*) yang optimal. Tujuannya adalah meminimalkan penggunaan memori puncak `M_peak` dengan mengorbankan komputasi ulang. Untuk grafik komputasi dengan `N` node, pemilihan subset `S` node untuk disimpan (*checkpoint*) merupakan masalah optimasi:
\[
\text{Minimalkan } M\_peak(S) \quad \text{dengan kendala } T\_recomputation(S) \leq (1 + \lambda) \cdot T\_baseline
\]
`T_baseline` adalah waktu tanpa *checkpointing*, dan `λ` (lambda) adalah faktor toleransi penambahan waktu (misalnya, `0.2` untuk 20% lebih lambat). Zenith dapat menggunakan algoritma heuristik (seperti dalam Chen et al., 2016) untuk menyelesaikan ini secara dinamis berdasarkan profil memori.

**3. Kuantisasi dengan Kalibrasi Kesadaran Distribusi:**
Proses kuantisasi Zenith ke INT8 tidak hanya menggunakan skala (*scale*) dan *zero-point* statis. Ia melakukan kalibrasi berbasis distribusi input aktual. Untuk tensor `X`, ia mencari parameter kuantisasi `s` (skala) dan `z` (*zero-point*) yang meminimalkan kesalahan kuantisasi:
\[
\min_{s, z} \| X - s \cdot (\text{quantize}(X/s + z) - z) \|^2
\]
Proses kalibrasi dapat menggunakan metode seperti *entropy minimization* (meminimalkan divergensi KL antara distribusi float dan kuantisasi) atau *percentile matching* (misalnya, memetakan persentil ke-99.99% dari `X` ke nilai maksimum INT8). Metode ini lebih unggul daripada `min-max` sederhana dan telah digunakan dalam *framework* seperti TensorRT dan NVIDIA's TensorFlow Quantization Toolkit.

#### **4.3. Spesifikasi Bahasa Pemrograman dan Framework Pendukung**
Pilihan teknologi didorong oleh kebutuhan akan performa, portabilitas, dan integrasi ekosistem.

**1. Bahasa Pemrograman Inti (Core Language):**
- **C++20/23**: Dipilih sebagai bahasa utama untuk *core engine* karena:
    - **Kontrol Performa dan Memori Mutlak**: Memungkinkan manajemen memori manual, alokasi yang selaras dengan cache, dan penggunaan intrinsik SIMD (AVX-512, NEON) secara langsung.
    - **Akses ke API Perangkat Keras Native**: Dapat memanggil *driver* CUDA, ROCm, oneAPI, dan Metal secara langsung dengan overhead minimal.
    - **Polimorfisme Waktu Kompilasi**: Menggunakan template dan konsep C++20 untuk membuat kode generik yang dioptimasi untuk tipe data dan arsitektur berbeda tanpa overhead *runtime*.
    - **Ekosistem yang Matang**: *Library* seperti Eigen (aljabar linear), fmt (formatting), dan spdlog (logging) dapat diintegrasikan.
- **Rust (Sebagai Pilihan Alternatif/Komoditas Tertentu)**: Dipertimbangkan untuk komponen yang memerlukan jaminan keamanan memori (*memory safety*) tanpa pengumpulan sampah (*GC*), seperti manajer memori lintas-perangkat atau komponen *network serving*. Namun, C++ tetap menjadi pilihan utama karena ekosistem AI yang lebih luas.

**2. *Python Bindings* dan API Pengguna:**
- **PyBind11**: Digunakan untuk membuat *binding* Python ke kode C++ inti. Alat ini memungkinkan ekspos API yang *idiomatic* ke Python dengan overhead yang sangat rendah dan mendukung konversi otomatis objek seperti `numpy.ndarray` ↔ `zenith::Tensor`.
- **API Python Zenith** akan dirancang minimalis dan eksplisit:
    ```python
    import zenith
    import torch

    # Model dari PyTorch
    model = torchvision.models.resnet50()
    
    # Optimisasi dengan konfigurasi jelas
    optimized_model = zenith.compile(
        model=model,
        target="cuda:0",  # atau "roc:0", "cpu", "tpu"
        precision="fp16",
        opt_level=3,      # Level agresivitas optimasi (1-3)
        tolerance=1e-5    # Batas kesalahan numerik
    )
    
    # Model yang dioptimalkan dapat digunakan seperti model asli
    output = optimized_model(input_tensor)
    ```

**3. *Framework* dan Teknologi Pendukung Kritis:**
- **ONNX (Open Neural Network Exchange)**: Berfungsi sebagai **format representasi menengah (IR) universal** dalam pipeline Zenith. Semua *adapter* (*frontend*) akan mengonversi model ke grafik ONNX. *Compiler* inti Zenith kemudian akan mengonsumsi dan mengoptimasi grafik ONNX ini. Ini memungkinkan dukungan untuk banyak *framework* sumber tanpa menulis *parser* khusus untuk masing-masing.
- **MLIR (Multi-Level Intermediate Representation)**: Dipertimbangkan sebagai **IR internal yang lebih canggih** untuk tahap optimisasi Zenith. MLIR memungkinkan representasi komputasi dari level tinggi (grafik operasi) hingga level rendah (loop, vektor) dalam satu *framework*, yang ideal untuk transformasi dan *lowering* yang kompleks ke berbagai *backend*.
- **oneDNN (Intel), cuDNN/cuBLAS (NVIDIA), rocBLAS/hipBLAS (AMD)**: Zenith tidak akan menulis ulang *kernel* matematika tingkat rendah. Sebaliknya, ia akan bertindak sebagai **pengelola dan pengirim (*dispatcher*) cerdas** untuk *library* yang sudah sangat dioptimasi ini. Lapisan abstraksi perangkat keras Zenith akan memanggil *primitive library* yang sesuai.

#### **4.4. Diagram Kelas (Class Diagram) Inti**
Diagram berikut menunjukkan hubungan antara kelas-kelas utama dalam *Core Engine* Zenith.

```
┌─────────────────────────────────────────────────────────────┐
│                      zen::CompilationSession                │
│  - model_ir: unique_ptr<GraphIR>                            │
│  - target_desc: TargetDescriptor                            │
│  + compile(source_model): Status                            │
│  + optimize(passes): Status                                 │
│  + get_compiled_artifact(): CompiledArtifact                │
└───────────────────────┬─────────────────────────────────────┘
                        │ uses
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                         zen::GraphIR                        │
│  - nodes: vector<unique_ptr<Node>>                          │
│  - edges: vector<Edge>                                      │
│  + add_node(op_type, inputs): Node*                         │
│  + fuse_pattern(pattern): bool                              │
│  + lower_to_target(target): unique_ptr<LoweredGraph>        │
└───────────────────────┬─────────────────────────────────────┘
                        │ contains
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                         zen::Node                           │
│  - op_type: string                                          │
│  - attrs: map<string, Attribute>                            │
│  - inputs: vector<TensorDescriptor>                         │
│  - outputs: vector<TensorDescriptor>                        │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────┐        ┌──────────────┐
│ zen::Op │        │ zen::Tensor  │
│ (Konkrit)│       │ Descriptor   │
└─────────┘        └──────────────┘
(Conv, MatMul, ...)
       │
       │ implements
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    zen::KernelRegistry                      │
│  - kernels: map<OpSignature, unique_ptr<Kernel>>            │
│  + register_kernel(signature, kernel): void                 │
│  + dispatch(op, target, device): Kernel*                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     zen::HardwareBackend                    │
│  # device_id: int                                           │
│  + alloc_memory(size): void*                                │
│  + launch_kernel(kernel, args): void                        │
│  + synchronize(): void                                      │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│ zen::CUDABackend│    │ zen::ROCMBackend │
└─────────────┘    └─────────────┘
```

**Penjelasan Kelas Kunci:**

*   **`CompilationSession`**: Kelas utama yang mengoordinasikan seluruh proses kompilasi. Pengguna membuat sesi, memberi makan model, dan meminta artefak yang dikompilasi.
*   **`GraphIR`**: Merepresentasikan grafik komputasi model dalam memori. Menyediakan metode untuk transformasi seperti fusi.
*   **`Node` & **`Op`**: `Node` adalah simpul dalam grafik, yang berisi metadata. Ia mengacu pada objek `Op` konkret (seperti `ConvOp`) yang memiliki logika eksekusi.
*   **`KernelRegistry`**: Pencatat (*registry*) global yang memetakan tanda tangan operasi (`OpSignature`—mencakup tipe operasi, tipe data, bentuk) ke implementasi `Kernel` yang tersedia. Ini adalah inti dari sistem *dispatch*.
*   **`HardwareBackend`**: Kelas dasar abstrak untuk semua *backend* (CUDA, ROCm, CPU). Kelas turunannya mengimplementasikan operasi spesifik perangkat keras.

Bab ini telah membangun fondasi teknis dan matematika yang konkret untuk Zenith. Bab selanjutnya akan membahas bagaimana fondasi ini diwujudkan: strategi implementasi bertahap, filosofi pengujian yang ketat, dan integrasi ke dalam *pipeline* pengembangan perangkat lunak modern.

### **BAB 5: STRATEGI IMPLEMENTASI DAN PENGUJIAN**

Bab ini menguraikan pendekatan sistematis untuk mewujudkan Zenith dari cetak biru menjadi perangkat lunak yang dapat diandalkan dan siap produksi. Strategi ini dibangun berdasarkan prinsip "Production-Ready from Day One" dan mengintegrasikan metodologi pengembangan yang ketat dengan siklus pengujian yang komprehensif.

#### **5.1. Kerangka Implementasi 6-Fase**
Proyek Zenith akan dijalankan dalam enam fase berurutan, di mana setiap fase harus mencapai serangkaian milestone yang terukur dan terdokumentasi dengan baik sebelum fase berikutnya dimulai. Pendekatan ini meminimalkan risiko teknis dan memastikan kualitas tertanam sejak awal.

**Fase 0: Persiapan dan Desain Mendalam (Bulan 1-6)**
Sebelum satu baris kode pun ditulis, fase ini didedikasikan untuk membangun fondasi yang kokoh.
*   **Kegiatan Utama**:
    1.  **Penyusunan Spesifikasi Teknis Detil (TRD)**: Setiap komponen dari Bab 3 dan 4 akan diuraikan menjadi dokumen spesifikasi teknis berbahasa Inggris yang mendetail, mencakup API, struktur data, algoritma, dan interaksi.
    2.  **Proof-of-Concept (PoC) Matematika dan Algoritma**: Implementasi skala kecil dari rumus optimasi inti (seperti kuantisasi kalibrasi, algoritma fusi) dalam Python/NumPy murni untuk memvalidasi kebenaran matematika dan keluaran sebelum diimplementasikan dalam C++.
    3.  **Desain Arsitektur Perangkat Lunak (Software Architecture Document - SAD)**: Dokumen yang mendefinisikan modul, antarmuka, dependensi, dan pola komunikasi, dilengkapi dengan diagram UML lengkap (sequence, component).
    4.  **Pembuatan Prototipe *Adapter* Sederhana**: Prototipe *adapter* ONNX (menggunakan `onnxruntime`) dan *adapter* PyTorch (menggunakan `torch.onnx.export`) untuk memvalidasi alur konversi model ke IR.
*   **Kriteria Kelulusan**: Semua dokumen desain direview dan disetujui oleh tim arsitek utama; semua PoC memberikan hasil yang sesuai dengan ekspektasi teoritis dan benchmark awal.

**Fase 1: Pembangunan *Core Engine* dan Abstraksi Dasar (Bulan 7-12)**
Fokus pada implementasi inti sistem dalam C++ modern.
*   **Kegiatan Utama**:
    1.  **Implementasi `GraphIR` dan Sistem Node**: Membangun struktur data dalam memori untuk grafik komputasi yang efisien, beserta *pass* optimasi dasar (*dead code elimination*, *constant folding*).
    2.  **Implementasi *Hardware Abstraction Layer (HAL)***: Membangun kelas dasar `HardwareBackend` dan implementasi konkrit pertama untuk **Backend CPU** (menggunakan SIMD intrinsik: AVX2/AVX-512 untuk x86, NEON untuk ARM) dan **Backend CUDA**.
    3.  **Pembangun *KernelLibrary* Dasar**: Implementasi sejumlah kecil kernel operasi fundamental (MatMul, Conv2D, ReLU) untuk backend CPU dan CUDA.
    4.  **Pengikatan (*Binding*) Python Awal dengan PyBind11**: Membuat *binding* minimal untuk menguji *engine* dari Python.
*   **Kriteria Kelulusan**: *Core engine* dapat memuat grafik ONNX sederhana, melakukan beberapa *pass* optimasi, dan mengeksekusinya dengan hasil numerik yang benar (memenuhi batas `δ`) di CPU dan GPU NVIDIA.

**Fase 2: Pengembangan *Framework Adapter* dan Optimasi Lanjutan (Bulan 13-18)**
Memperluas kompatibilitas dan kecerdasan sistem.
*   **Kegiatan Utama**:
    1.  **Penyempurnaan *Adapter* Produksi**: Pengembangan *adapter* yang robust untuk PyTorch (`torch.export`), TensorFlow (`tf.saved_model`), dan JAX (melalui `jax2onnx` atau fungsi ekspor kustom).
    2.  **Implementasi *Optimization Passes* Canggih**: Logika untuk *operator fusion* (Conv-BN-ReLU, GeMM-Add), *layout transformation* (NHWC ke NCHW), dan *kernel auto-tuning* sederhana.
    3.  **Penambahan Backend Lain**: Implementasi **Backend ROCm** (untuk GPU AMD) dan **Backend oneAPI** (untuk GPU Intel).
    4.  **Pengembangan Sistem Profil dan *Benchmark* Internal**: Alat untuk mengukur kinerja setiap tahap dan operasi secara terperinci.
*   **Kriteria Kelulusan**: Zenith dapat mengompilasi dan mempercepat model standar (misalnya, ResNet-50, BERT-base) dari ketiga *framework* sumber, dan menunjukkan peningkatan kecepatan inferensi yang terukur (>1.5x) di atas eksekusi *framework* native pada kasus-kasus tertentu.

**Fase 3: Implementasi Teknik Presisi Campuran dan Kuantisasi (Bulan 19-24)**
Menambahkan kemampuan optimasi yang lebih agresif.
*   **Kegiatan Utama**:
    1.  **Manajemen Presisi Campuran Otomatis**: Implementasi analisis rentang dinamis dan *safety margin* seperti dirumuskan di Bab 4.1, mendukung FP16 dan BF16.
    2.  **Pipeline Kuantisasi INT8 Penuh**: Implementasi kalibrasi statis dan dinamis, *quantize-aware training* (QAT) *simulation*, dan kernel terkuantisasi untuk backend yang mendukung (CPU dengan VNNI, GPU dengan Tensor Cores/Matrix Cores).
    3.  **Mekanisme *Fallback* dan Validasi Numerik**: Sistem yang secara otomatis mendeteksi overflow/underflow dan *fallback* ke presisi lebih tinggi, disertai validasi kesalahan otomatis terhadap model referensi.
*   **Kriteria Kelulusan**: Zenith dapat mengonversi model FP32 ke FP16/INT8 dengan kehilangan akurasi yang kurang dari batas yang ditentukan (`tolerance`), dan masih mengeksekusinya dengan benar di backend yang ditargetkan.

**Fase 4: Pengujian Ekstensif, *Hardening*, dan Dokumentasi (Bulan 25-30)**
Fase penjaminan kualitas dan kesiapan produksi.
*   **Kegiatan Utama**:
    1.  **Pengujian Skala Besar**: Menjalankan seluruh *suite* pengujian (lihat 5.3) pada berbagai konfigurasi perangkat keras dan perangkat lunak.
    2.  ***Fuzz Testing* dan *Chaos Engineering***: Menyuntikkan kesalahan dan data acak untuk menguji ketangguhan sistem.
    3.  **Audit Keamanan dan Analisis Kode Statis**: Menggunakan alat seperti `clang-tidy`, `Coverity`, atau `CodeQL`.
    4.  **Penulisan Dokumentasi Komprehensif**: Dokumentasi API, panduan pengguna, tutorial, dan *whitepaper* teknis.
*   **Kriteria Kelulusan**: Tidak ada *bug* kritis (*critical*) atau mayor (*major*) yang terbuka; cakupan kode (*code coverage*) >95%; dokumentasi lengkap untuk semua fitur yang diimplementasikan.

**Fase 5: Rilis Awal, Umpan Balik Komunitas, dan Penyempurnaan (Bulan 31-36)**
Meluncurkan produk ke komunitas terbatas dan mengiterasinya.
*   **Kegiatan Utama**:
    1.  **Rilis Alpha dan Beta Terbatas**: kepada mitra riset dan perusahaan terpilih.
    2.  **Pembentukan Komunitas dan *Issue Tracking***: Mempersiapkan repositori GitHub, *code of conduct*, dan *contribution guidelines*.
    3.  **Pengumpulan Umpan Balik dan *Benchmark* Independen**: Membiarkan komunitas menguji Zenith dalam skenario dunia nyata.
    4.  **Iterasi Cepat Berdasarkan Umpan Balik**: Memperbaiki *bug* dan menyesuaikan fitur berdasarkan masukan.
*   **Kriteria Kelulusan**: Rilis stabil pertama (v1.0.0) kepada publik.

#### **5.2. Strategi Pengembangan Perangkat Lunak**
Pembangunan Zenith akan mengikuti praktik rekayasa perangkat lunak modern yang ketat.

*   **Kontrol Versi dan *Branching Strategy***: Menggunakan Git dengan model alur GitFlow yang dimodifikasi. `main` selalu dalam keadaan siap rilis. `develop` adalah cabang integrasi. Fitur baru dikembangkan di cabang `feature/`. Rilis menggunakan cabang `release/`. Perbaikan cepat (*hotfix*) menggunakan cabang `hotfix/`.
*   ***Code Review* Wajib dan Berpasangan (*Pair Programming*)***: Tidak ada kode yang dapat digabungkan (*merge*) ke `develop` atau `main` tanpa melalui *pull request* (PR) yang disetujui oleh minimal dua *reviewer* selain penulis. *Pair programming* akan digunakan untuk modul inti yang kompleks.
*   ***Style Guide* dan *Static Analysis***:
    *   **C++**: Mengikuti *Google C++ Style Guide* dengan penyesuaian tertentu. Format kode otomatis dengan `clang-format`. Analisis statis dengan `clang-tidy`.
    *   **Python**: Mengikuti PEP 8. Format kode otomatis dengan `black`. Pemeriksaan kualitas dengan `pylint` dan `mypy` (untuk *type hints*).
    *   ***Commit Message Convention***: Mengikuti *Conventional Commits* (contoh: `feat(hal): add initial ROCm backend support`, `fix(core): segfault in GraphIR destructor`).
*   ***Continuous Integration (CI) Dasar***: Setiap PR akan memicu *pipeline* CI (misalnya, GitHub Actions) yang menjalankan:
    1.  Pemeriksaan gaya kode (*linter*).
    2.  Kompilasi untuk beberapa platform (Linux GCC/Clang, Windows MSVC).
    3.  *Suite* pengujian unit dan integrasi cepat.
*   **Manajemen Ketergantungan (*Dependency Management*)**: Menggunakan `vcpkg` atau `conan` untuk mengelola *library* C++ pihak ketiga (seperti Protobuf, fmt, pybind11). *Dependency* Python dikelola dengan `poetry` atau `uv`.

#### **5.3. Strategi Pengujian (Testing) Menyeluruh**
Pyramid pengujian Zenith akan sangat luas dan dalam, mencakup lebih dari sekadar pengujian fungsional.

**1. Pengujian Unit (Unit Testing) – Dasar Piramida**
*   **Cakupan**: Setiap kelas dan fungsi C++ inti (`GraphIR`, `Kernel`, `Backend`) serta fungsi Python *wrapper*.
*   ***Framework***: **GoogleTest** (C++), **pytest** (Python).
*   **Sasaran**: Memastikan kebenaran logis setiap unit terkecil. Mocking digunakan untuk mengisolasi dependensi eksternal (seperti *driver* GPU).
*   ***Code Coverage***: Ditargetkan >95% untuk kode C++ inti, diukur dengan `gcov`/`llvm-cov`.

**2. Pengujian Integrasi (Integration Testing)**
*   **Cakupan**: Interaksi antar modul utama (misalnya, `Adapter -> GraphIR -> Optimizer -> Backend`).
*   **Kegiatan**:
    *   **Uji Kompilasi End-to-End Sederhana**: Memastikan alur dari model ONNX hingga eksekusi berjalan tanpa error.
    *   **Uji Kebenaran Numerik (*Numerical Correctness*)**: Membandingkan keluaran model yang dioptimasi Zenith dengan keluaran referensi dari *framework* asli (PyTorch/TF) untuk berbagai jenis input, memverifikasi batas kesalahan `δ` terpenuhi.

**3. Pengujian Kinerja dan Regresi (Performance & Regression Testing)**
*   **Cakupan**: Seluruh *pipeline* optimasi pada model dan konfigurasi *hardware* yang representatif.
*   **Kegiatan**:
    *   ***Benchmark* Harian (*Nightly Benchmarks*)**: *Pipeline* otomatis yang menjalankan kumpulan model (*ResNet, BERT, GPT-2*) di berbagai *backend* dan konfigurasi presisi. Hasilnya dibandingkan dengan *baseline* (kinerja *framework* native) dan versi Zenith sebelumnya.
    *   ***Performance Regression Gate***: Jika suatu komitmen menyebabkan penurunan kinerja >5% untuk *benchmark* kunci, *pipeline* CI gagal.

**4. Pengujian Sistem dan Penerimaan (System & Acceptance Testing)**
*   **Cakupan**: Perilaku sistem secara keseluruhan, termasuk antarmuka pengguna.
*   **Kegiatan**:
    *   **Uji Beban (*Load Testing*)**: Mengukur kinerja dan penggunaan memori di bawah *throughput* inferensi yang tinggi.
    *   **Uji Ketahanan (*Soak Testing*)**: Menjalankan *server* inferensi Zenith terus-menerus selama 72+ jam untuk mendeteksi kebocoran memori atau *resource leak*.

**5. Pengujian Khusus dan Lanjutan**
*   **Pengujian Kekacauan (*Chaos Testing*)**: Menyuntikkan kegagalan ke dalam sistem (misalnya, membunuh proses *driver* GPU, mensimulasikan kehabisan memori) untuk memverifikasi bahwa Zenith gagal dengan baik (*graceful degradation*) atau pulih.
*   **Pengujian Mutasi (*Mutation Testing*)**: Menggunakan alat seperti **Mutation++** (C++) untuk menyuntikkan *bug* kecil ke dalam kode sumber yang sudah diuji, lalu menjalankan *suite* pengujian unit. Jika pengujian masih lulus, berarti *test suite* tidak cukup kuat. Target adalah *mutation score* >90%.
*   **Pengujian Properti (*Property-Based Testing*) – Khusus untuk Matematika**: Menggunakan **Hypothesis** (Python) untuk menghasilkan ratusan atau ribuan input acak ke fungsi transformasi matematika (misalnya, kuantisasi), dan memverifikasi bahwa properti yang diinginkan (misalnya, batas kesalahan) selalu terpenuhi.

**6. Pengujian Kompatibilitas dan *Cross-Platform***
*   **Cakupan**: Berbagai kombinasi OS (Linux, Windows), versi *driver* GPU, dan versi *framework* sumber.
*   ***Strategy***: Menggunakan matriks CI dan *farm* mesin uji fisik/virtual dengan konfigurasi yang berbeda-beda.

#### **5.4. Integrasi ke dalam CI/CD Pipeline**
Semua strategi pengujian di atas akan diotomatisasi dan diintegrasikan ke dalam *pipeline* CI/CD yang bertingkat.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline CI/CD Zenith                       │
├─────────────────────────────────────────────────────────────────┤
│  Trigger: Setiap Pull Request ke `develop`                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     Tahap 1: Build & Lint                 │  │
│  │  - C++ Build (Linux GCC/Clang, Windows MSVC)              │  │
│  │  - Python Package Build                                   │  │
│  │  - clang-format / black / pylint                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               Tahap 2: Pengujian Cepat (Fast)             │  │
│  │  - Unit Tests (CPU-only)                                  │  │
│  │  - Integration Tests (CPU-only)                           │  │
│  │  - Numerical Correctness Tests (skala kecil)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Trigger: Setiap Penggabungan ke `develop` / Setiap Malam       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            Tahap 3: Pengujian Panjang (Extended)          │  │
│  │  - Unit & Integration Tests (dengan GPU)                  │  │
│  │  - Performance Regression Suite (subset benchmark)        │  │
│  │  - Compatibility Tests (versi ONNX/PyTorch berbeda)       │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Tahap 4: Pengujian Periodik                  │  │
│  │  (Jalankan 1x per minggu atau sebelum rilis besar)        │  │
│  │  - Full Benchmark Suite (semua model, semua backend)      │  │
│  │  - Chaos Testing Sessions                                 │  │
│  │  - Soak/Long-Running Tests                                │  │
│  │  - Mutation Testing                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

*   ***Pipeline* untuk Rilis**: Ketika cabang `release/` dibuat, *pipeline* khusus akan dijalankan yang mencakup **semua** tahap pengujian, termasuk audit keamanan dan pembangunan paket distribusi (*Docker image*, *Python wheel*, *binary installer*).

Dengan strategi implementasi dan pengujian yang ketat ini, Zenith tidak hanya dirancang untuk menjadi unggul secara teori, tetapi juga dibangun untuk menjadi kuat, andal, dan dapat dipercaya dalam praktiknya. Bab selanjutnya akan membahas bagaimana Zenith berinteraksi dengan ekosistem yang sudah ada. 

### **BAB 6: INTEGRASI DENGAN EKOSISTEM YANG ADA**

Kekuatan utama Zenith bukan hanya pada algoritmanya yang canggih, tetapi pada kemampuannya untuk tertanam secara mulus ke dalam ekosistem *Machine Learning* (ML) yang telah mapan dan sangat luas. Bab ini menjelaskan strategi teknis untuk mencapai integrasi yang dalam dengan *framework* populer, format pertukaran, berbagai perangkat keras, dan alat-alat pendukung (*tooling*) komunitas.

#### **6.1. Mekanisme Integrasi dengan PyTorch, TensorFlow, dan JAX**
Zenith mengambil pendekatan *non-invasif* dengan bertindak sebagai *backend compiler* tambahan yang dapat "dipanggil" oleh *framework* sumber. Ini berbeda dengan mencoba membungkus atau mengganti *runtime* asli mereka.

**1. Integrasi dengan PyTorch (`torch.compile` sebagai *Hook*):**
Pendekatan yang paling elegan adalah memanfaatkan infrastruktur kompilasi baru PyTorch, `torch.compile` (dikenal sebagai **TorchDynamo** dan **TorchInductor**). Zenith dapat diimplementasikan sebagai ***custom backend*** untuk `torch.compile`.
```python
import torch
import zenith

# Daftarkan Zenith sebagai backend untuk torch.compile
torch._dynamo.reset()
zenith_backend = zenith.torch.create_backend(device="cuda", precision="fp16")
torch.compile(backend=zenith_backend)

# Gunakan decorator pada model atau fungsi
model = torchvision.models.resnet50()
optimized_model = torch.compile(model)

# Eksekusi akan dirutekan melalui Zenith
output = optimized_model(input_tensor)
```
**Cara Kerja**: Saat `torch.compile` dipanggil, **TorchDynamo** akan menangkap *graph* dari model (*FX Graph*). Alih-alih mengirim *graph* ini ke *backend* default PyTorch (Inductor), *graph* tersebut diteruskan ke *adapter* Zenith. *Adapter* ini mengonversi *FX Graph* ke **ONNX** (atau langsung ke *GraphIR* Zenith), kemudian diproses oleh *engine* Zenith, dan menghasilkan *kernel* yang dikompilasi. Pada eksekusi, PyTorch hanya memanggil *kernel* Zenith yang sudah jadi.

**2. Integrasi dengan TensorFlow (melalui `tf.function` dan *Custom Ops*):**
Untuk TensorFlow 2.x, Zenith dapat diintegrasikan melalui dua jalur:
*   ***Plugin untuk Grappler***: **Grappler** adalah *graph optimizer* default TensorFlow. Zenith dapat dikemas sebagai *plugin* Grappler yang mendaftarkan *optimization passes* kustom. Saat `tf.function(experimental_compile=True)` dipanggil, *graph* TensorFlow dapat diproses oleh *pass* Zenith sebelum dikirim ke XLA atau *runtime* lain.
*   ***TensorFlow Operator (Op) Kustom***: Untuk operasi yang sangat terspesialisasi (misalnya, *kernel fusion* eksotis), Zenith dapat mengimplementasikannya sebagai **TensorFlow Custom Op** dalam C++/CUDA. *Op* ini kemudian dapat dipanggil dari kode Python TF, tetapi logika intinya dieksekusi oleh *runtime* Zenith.

**3. Integrasi dengan JAX (Sebagai *External Callable*):**
Karena JAX dibangun di atas fungsi murni dan XLA, integrasi dapat dilakukan dengan memperlakukan sub-grafik yang dioptimasi Zenith sebagai ***external callable***.
```python
import jax
import jax.numpy as jnp
import zenith

# Kompilasi sebuah fungsi JAX dengan XLA seperti biasa
def pure_function(x):
    return jnp.dot(x, x.T)

# Buat fungsi yang bagian dalamnya dioptimasi Zenith
def hybrid_function(x):
    # Konversi bagian komputasi ke format ONNX (contoh)
    onnx_graph = zenith.jax.export_subgraph(pure_function, x)
    # Optimasi subgraph tersebut dengan Zenith
    optimized_subgraph = zenith.compile(onnx_graph, target="tpu")
    # Kembalikan sebagai fungsi yang dapat dipanggil JAX
    return zenith.jax.as_jax_callable(optimized_subgraph)
```
Pendekatan ini memungkinkan peneliti untuk secara manual mengidentifikasi *bottleneck* dalam kode JAX mereka dan menyerahkannya ke Zenith untuk optimasi lebih agresif, sambil mempertahankan alur kerja JAX yang tersisa.

#### **6.2. Peran ONNX sebagai Format Pertukaran Netral**
**ONNX** bukan hanya *one of many* fitur, melainkan **strategic cornerstone** (landasan strategis) dari arsitektur Zenith. Ia berfungsi sebagai *lingua franca* atau bahasa pemersatu.

*   ***Single Conversion Target***: Semua *framework adapter* (PyTorch, TensorFlow, JAX) memiliki tugas utama yang sama: mengonversi model dari format asli ke **ONNX**. Ini menyederhanakan arsitektur *frontend* Zenith secara signifikan. Alih-alih menulis parser untuk setiap format *framework*, Zenith hanya perlu memelihara satu *pipeline* pemrosesan ONNX yang sangat robust.
*   ***Rich Operator Set and Versioning***: ONNX mendefinisikan sekumpulan operator standar yang terus berkembang. Zenith dapat fokus mengoptimasi implementasi untuk operator ONNX ini. Dukungan versi ONNX yang ketat memastikan kompatibilitas ke belakang.
*   ***Validation and Sanitization***: *ONNX Runtime* sendiri menyediakan alat seperti `onnx.checker` dan `onnx.shape_inference`. Zenith dapat memanfaatkannya untuk memvalidasi dan membersihkan model sebelum proses optimasi internal, memastikan input yang valid ke *core engine*.
*   ***Ecosystem Leverage***: Dengan menggunakan ONNX, Zenith secara otomatis mendapatkan kompatibilitas dengan ratusan model yang sudah ada di **Hugging Face Model Hub**, **ONNX Model Zoo**, dan repositori lainnya yang sudah menyediakan ekspor ONNX.

**Diagram Alur dengan ONNX:**
```
[PyTorch Model] → [torch.onnx.export] → [ONNX Model]
[TF SavedModel] → [tf2onnx.convert]   → [ONNX Model] → [Zenith Core Optimizer] → [Optimized Binary]
[JAX Function]  → [jax2onnx.export]   → [ONNX Model]
```

#### **6.3. Strategi Abstraksi untuk Berbagai Perangkat Keras (CPU, GPU, TPU, NPU, FPGA)**
Filosofi "Write Once, Run Anywhere" Zenith diwujudkan melalui **Hardware Abstraction Layer (HAL)** yang terinspirasi oleh prinsip-prinsip *portable computing* seperti **SYCL** dan **Vulkan**.

**1. Arsitektur HAL Berlapis:**
```
┌─────────────────────────────────────────────┐
│         Zenith Kernel Library (ZenKL)       │ ← Operasi high-level (MatMul, Conv)
│     (Target-Independent Graph IR)           │
├─────────────────────────────────────────────┤
│      Runtime Dispatch & Memory Manager      │ ← Pilih backend, kelola memori terpadu
├─────────────────────────────────────────────┤
│  Backend Interface (Pure Virtual Class)     │ ← Kontrak API untuk semua backend
├──────┬──────┬───────┬──────┬──────┬───────┤
│ CUDA │ ROCm │ oneAPI│ Metal│ Vulkan│ CPU   │ ← Implementasi backend spesifik
│Backend│Backend│Backend│Backend│Backend │Backend│
└──────┴──────┴───────┴──────┴──────┴───────┘
```

**2. Contoh Implementasi `Backend` Interface (Sederhana):**
```cpp
class HardwareBackend {
public:
    virtual ~HardwareBackend() = default;
    // Manajemen Memori
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t bytes, CopyDirection dir) = 0;
    
    // Kompilasi dan Eksekusi Kernel
    virtual KernelHandle compile_kernel(const std::string& kernel_source, const std::string& kernel_name) = 0;
    virtual void launch_kernel(KernelHandle kernel, const LaunchParams& params, void** args) = 0;
    
    // Sinkronisasi
    virtual void synchronize() = 0;
    
    // Query Kemampuan
    virtual DeviceCapabilities get_capabilities() const = 0;
};
```

**3. Strategi untuk Berbagai Tipe Perangkat Keras:**
*   **CPU (x86/ARM)**: Menggunakan **oneDNN** (Intel) atau **ACL** (ARM Compute Library) untuk kernel yang telah dioptimasi. Untuk operasi generik, menghasilkan kode LLVM IR dan mengompilasinya secara *just-in-time* (JIT) untuk CPU target dengan dukungan instruksi spesifik (AVX-512, NEON, SVE).
*   **GPU NVIDIA (CUDA)**: Menggunakan **cuDNN**, **cuBLAS**, dan **cuDNN** untuk operasi primitif. Untuk *kernel fusion* kustom, menghasilkan kode **PTX** atau **CUBIN** menggunakan compiler NVCC atau NVRTC (*Runtime Compilation*).
*   **GPU AMD (ROCm)**: Menggunakan **rocBLAS**, **MIOpen**. Mengikuti pola yang sama dengan CUDA tetapi dengan toolchain HIP. Zenith dapat mengkompilasi *kernel* dari sumber HIP atau IR SPIR-V.
*   **Google TPU**: Berintegrasi dengan **XLA/TPU Runtime**. Zenith akan mengonversi *GraphIR*-nya menjadi **HLO (High-Level Optimizer)** yang merupakan IR XLA, dan membiarkan kompiler XLA Google yang matang melakukan *lowering* ke kode TPU.
*   **NPU/Accelerator Lain (Intel Gaudi, Habana, dll.)**: Untuk akselerator dengan *runtime* dan driver khusus, HAL akan mengimplementasikan *backend* yang melakukan konversi dari *GraphIR* Zenith ke graf yang dimengerti oleh *runtime* proprietary tersebut, seringkali melalui lapisan API C yang disediakan vendor.

**4. *Auto-Tuning* Berbasis *Hardware*:**
Setiap *backend* akan dilengkapi dengan *database* profil berisi parameter *kernel* yang optimal (misalnya, ukuran *thread block*, *tiling size*) untuk berbagai operasi dan bentuk (*shape*) tensor pada perangkat keras spesifik tersebut. Pada *first run*, Zenith dapat menjalankan *micro-benchmark* untuk mengisi *database* ini, memungkinkan performa optimal di berbagai perangkat.

#### **6.4. Kompatibilitas dengan Alat dan Platform Lain**
Zenith bertujuan untuk menjadi warga negara yang baik di ekosistem MLOps.

*   **Integrasi dengan Hugging Face `transformers`**:
    Zenith dapat menyediakan *pipeline* optimasi otomatis untuk model Hugging Face.
    ```python
    from transformers import AutoModelForSequenceClassification
    import zenith
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    optimized_model = zenith.optimize_transformers_model(model, target="cpu", precision="int8")
    # optimized_model dapat digunakan seperti model HF biasa
    ```
*   ***Inference Server* (TensorFlow Serving, Triton)**:
    Zenith dapat dikemas sebagai ***backend*** untuk **NVIDIA Triton Inference Server**. Dengan membuat *backend* Triton bernama "zenith", model yang dikompilasi Zenith dapat disajikan dengan skala tinggi, mendapatkan manfaat dari *dynamic batching*, *concurrent execution*, dan *monitoring* Triton.
*   ***Experiment Tracking* (Weights & Biases, MLflow)**:
    Zenith dapat mengintegrasikan *hook* untuk mencatat metrik pengoptimalan (seperti kecepatan sebelum/sesudah, penggunaan memori, perubahan akurasi) ke dalam *run* di **W&B** atau **MLflow**. Ini memberikan *trackability* yang lengkap dalam alur kerja MLOps.
*   ***Edge Deployment* (TensorFlow Lite, ONNX Runtime Mobile)**:
    Untuk perangkat *edge*, *compiler* Zenith dapat memiliki *target output* khusus, misalnya menghasilkan **TensorFlow Lite FlatBuffer** (`.tflite`) yang telah dioptimasi atau *binary* ONNX Runtime yang disesuaikan. Ini memungkinkan model yang dikembangkan di *framework* besar langsung jalan di perangkat *edge* dengan efisiensi Zenith.
*   **Dukungan untuk Scikit-Learn dan *Traditional ML***:
    Meskipun fokus pada *deep learning*, Zenith dapat menyediakan *adapter* untuk model **scikit-learn** (misalnya, *ensemble trees*, SVM) dengan mengonversinya ke dalam representasi grafik komputasi (mungkin melalui proyek seperti **sklearn-onnx**). Ini memperluas manfaat optimasi ke alur kerja ML yang lebih tradisional.

Dengan strategi integrasi yang luas dan dalam ini, Zenith tidak berada di dalam menara gading, tetapi menjadi pemain kunci yang menghubungkan dan memperkuat seluruh siklus hidup pengembangan AI—dari riset di PyTorch/JAX, melalui optimasi produksi, hingga deployment di cloud, *edge*, atau akselerator khusus.

### **BAB 7: NILAI DAN DAMPAK YANG DIHARAPKAN**

Bab ini menganalisis nilai strategis yang ditawarkan Zenith kepada berbagai pemangku kepentingan, memeriksa *trade-off* yang harus dikelola, dan memproyeksikan dampak potensialnya terhadap kecepatan inovasi dan efisiensi industri.

#### **7.1. Value Proposition bagi Berbagai Pemangku Kepentingan**
Nilai Zenith tidak monolithic; ia dirancang untuk menyelesaikan masalah spesifik bagi setiap segmen pengguna dalam ekosistem AI.

**1. Bagi Peneliti AI (AI Researcher):**
*   **Akselerasi Eksperimen Tanpa Perubahan Kode**: Peneliti dapat mempertahankan alur kerja fleksibel di PyTorch atau JAX, dan dengan menambahkan beberapa baris kode, mendapatkan akselerasi pelatihan dan evaluasi model yang signifikan (ditargetkan **1.5x hingga 3x**) di *hardware* yang mereka miliki. Ini berarti siklus eksperimen yang lebih cepat dan lebih banyak iterasi.
*   **Demokratisasi Akses ke Hardware Khusus**: Seorang peneliti di institusi dengan sumber daya terbatas dapat mengembangkan model di laptop berbasis CPU, lalu menggunakan Zenith untuk mengompilasinya secara optimal untuk GPU AMD di lab atau TPU yang disediakan cloud, **tanpa perlu mempelajari stack teknologi baru**. Ini mengurangi hambatan untuk bereksperimen dengan arsitektur besar.
*   **Reproduktibilitas yang Dipermudah**: Karena Zenith menjamin stabilitas numerik, seorang peneliti dapat membagikan `zenith_compile_config.yaml` bersama kode modelnya. Rekan lain dapat mereproduksi hasil eksak dengan kinerja optimal di lingkungan *hardware* yang berbeda, mengatasi masalah "works on my machine".

**2. Bagi Insinyur ML (ML Engineer) & Tim MLOps:**
*   **Penyederhanaan Alur Produksi dari Riset ke Deployment**: Zenith bertindak sebagai **jembatan standar**. Model dari tim riset (PyTorch) dapat secara otomatis dioptimasi dan dikonversi oleh Zenith ke dalam artefak berkinerja tinggi untuk *serving* produksi (misalnya, sebagai *backend* di **NVIDIA Triton** atau **TensorFlow Serving**). Ini menghilangkan tahap konversi *ad-hoc* yang rawan kesalahan.
*   ***Hardware Portability* dan Pengurangan Biaya**: Tim dapat membangun satu *pipeline* pelatihan/serving yang dapat berjalan di **berbagai jenis instance cloud** (GPU NVIDIA, GPU AMD, AWS Trainium/Inferentia) berdasarkan ketersediaan dan harga, tanpa menulis ulang kode. Ini memberikan fleksibilitas negosiasi dan ketahanan terhadap kekurangan pasokan *hardware* tertentu.
*   ***Performance Guarantee* yang Dapat Diukur**: Berbeda dengan *tuning* manual, Zenith memberikan laporan komprehensif yang memverifikasi bahwa optimasi tidak mengorbankan akurasi. Ini memberikan kepercayaan yang diperlukan untuk deployment ke produksi. Estimasi pengurangan biaya komputasi cloud ditargetkan sebesar **30-50%** untuk beban kerja inferensi melalui kombinasi kuantisasi dan *kernel fusion*.

**3. Bagi Vendor Hardware dan Penyedia Cloud:**
*   ***Lower Barrier to Adoption***: Vendor *hardware* baru (misalnya, NPU startup, vendor FPGA) menghadapi tantangan "chicken-and-egg": developer tidak menargetkan *hardware* mereka karena tidak ada dukungan *framework*, dan dukungan *framework* tidak dianggap prioritas karena tidak ada pengguna. Dengan mengimplementasikan **Zenith Backend** untuk *hardware* mereka, vendor langsung mendapatkan akses ke seluruh ekosistem model yang didukung Zenith, mendorong adopsi dengan lebih cepat.
*   ***Value-Add Service* bagi Cloud Provider**: Penyedia cloud (AWS, GCP, Azure) dapat menawarkan **Zenith sebagai layanan terkelola** yang diintegrasikan ke dalam platform ML mereka (seperti SageMaker, Vertex AI). Ini menjadi pembeda kompetitif yang memungkinkan pelanggan mendapatkan kinerja terbaik dari berbagai keluarga instance yang mereka tawarkan.
*   **Pengurangan Beban Pemeliharaan**: Alih-alih harus memelihara dan mengoptimasi *fork* atau *binding* khusus dari PyTorch/TensorFlow untuk setiap jenis akselerator mereka, mereka dapat berkontribusi pada satu *backend* Zenith yang terfokus.

**4. Bagi Komunitas Open Source Secara Keseluruhan:**
*   **Konsolidasi Upaya Optimasi**: Saat ini, upaya optimasi tersebar di berbagai *framework*, *compiler* (TVM, XLA, Glow), dan *library* vendor. Zenith bertujuan untuk menjadi **tempat berkumpulnya upaya optimasi lintas-platform**, di mana kontributor dari NVIDIA, Intel, Google, Meta, dan lainnya dapat bekerja sama pada satu kode basis untuk keuntungan semua orang.
*   **Peningkatan Kualitas dan Keandalan Kode AI**: Dengan fokus pada verifikasi matematis dan pengujian ekstensif, Zenith dapat menetapkan **standar baru untuk ketahanan kode sistem AI**. Praktik ini dapat mempengaruhi dan meningkatkan praktik pengembangan di proyek *upstream* (seperti PyTorch itu sendiri).

#### **7.2. Analisis Trade-off yang Dikelola secara Proaktif**
Tidak ada solusi yang sempurna. Zenith didesain untuk secara eksplisit mengakui dan mengelola *trade-off* berikut:

**1. Kecepatan vs. Generalisasi (*Speed vs. Generalization*):**
*   ***Trade-off***: Optimasi yang sangat agresif (misalnya, *auto-tuning* *kernel* untuk ukuran tensor tertentu) dapat menghasilkan kode yang sangat cepat untuk kasus tertentu, tetapi mungkin kurang optimal atau memerlukan *recompilation* untuk input dengan bentuk berbeda.
*   ***Strategi Zenith***: Menerapkan hierarki strategi. Tingkat optimasi `O1` menggunakan *kernel* yang digeneralisasi dengan baik. Tingkat `O3` dapat menghasilkan banyak varian *kernel* khusus dan menggunakan *runtime dispatch* berdasarkan bentuk input. Pengguna mengontrol *trade-off* ini melalui parameter `opt_level`.

**2. Waktu Kompilasi vs. Waktu Eksekusi (*Compilation Time vs. Execution Time*):**
*   ***Trade-off***: *Graph optimization*, *auto-tuning*, dan kompilasi JIT menambah *overhead* sebelum eksekusi pertama. Untuk skenario inferensi sekali pakai (*one-off*), ini merugikan.
*   ***Strategi Zenith***:
    1.  **Menyediakan Mode "Cached Compilation"**: Hasil kompilasi untuk sebuah model dan konfigurasi target dapat disimpan di *cache* dan digunakan kembali, menghilangkan *overhead* untuk run selanjutnya.
    2.  **Kompilasi Awal (*Ahead-of-Time/AOT*)** untuk Deployment: Di lingkungan produksi, kompilasi dilakukan sekali saat membangun *Docker image*, sehingga tidak ada *overhead* saat *runtime*.
    3.  **Kompromi melalui Profil**: Menggunakan profil statis dari bentuk input yang diketahui untuk mengurangi ruang pencarian *auto-tuning*.

**3. Fleksibilitas vs. Stabilitas (*Flexibility vs. Stability*):**
*   ***Trade-off***: Mendukung banyak *frontend* dan *backend* yang berkembang dengan cepat dapat membuat API internal kompleks dan rentan terhadap *breakage*.
*   ***Strategi Zenith***:
    1.  **Mengandalkan ONNX sebagai Batas Stabil**: ONNX menyediakan lapisan abstraksi yang relatif stabil antara *frontend* yang berubah-ubah dan *core engine*.
    2.  ***Backend Interface* yang Terdefinisi dengan Baik**: Kontrak HAL yang ketat mengisolasi perubahan di *backend* spesifik.
    3.  ***Extensive Testing Matrix***: CI/CD yang komprehensif memvalidasi kompatibilitas dengan berbagai versi *framework* dan *driver*.

**4. Kompleksitas Sistem vs. Kemudahan Penggunaan (*Complexity vs. Ease of Use*):**
*   ***Trade-off***: Sistem yang kuat secara inheren adalah kompleks.
*   ***Strategi Zenith***: Menyembunyikan kompleksitas melalui **API Python yang sangat sederhana** dan **konfigurasi yang masuk akal (*sensible defaults*)**. Kompleksitas penuh hanya diekspos kepada pengguna tingkat lanjut melalui API yang lebih granular.

#### **7.3. Dampak terhadap Kecepatan Riset dan Efisiensi Produksi**
Jika berhasil, Zenith memiliki potensi untuk menggeser paradigma dalam pengembangan dan deployment AI.

**1. Mempercepat Siklus Inovasi (Penelitian):**
*   **Pengurangan Waktu "Waiting on GPU"**: Percepatan pelatihan yang konsisten berarti peneliti mendapatkan hasil lebih cepat, menguji lebih banyak hipotesis.
*   ***Playground* Hardware yang Diperluas**: Peneliti dapat dengan mudah menguji apakah model mereka mendapat manfaat dari memori HBM yang besar pada GPU tertentu atau aritmatika tensor khusus pada TPU, mendorong eksplorasi arsitektur yang sebelumnya tidak praktis.
*   **Penelitian yang Dapat Direproduksi dengan Lebih Baik**: Menjembatani kesenjangan antara hasil penelitian (sering di PyTorch pada GPU tertentu) dan implementasi produksi dapat meningkatkan reproduktibilitas dan kemajuan kumulatif bidang ini.

**2. Meningkatkan Efisiensi dan Kelincahan Produksi:**
*   **Optimasi Sumber Daya Cloud**: Kemampuan untuk menjalankan beban kerja yang sama secara efisien di berbagai *hardware* memungkinkan perusahaan untuk **mengoptimalkan biaya cloud secara dinamis**, memilih instance yang memberikan performa/ harga terbaik pada waktu tertentu.
*   ***Future-proofing* Investensi Model**: Basis kode model yang ditulis untuk *framework* tinggi tidak lagi terkunci pada satu *vendor hardware*. Investasi dalam model AI dilindungi dari pergeseran lanskap *hardware*.
*   **Penyederhanaan Stack Teknologi**: Alih-alih memelihata keahlian dalam beberapa *runtime* dan alat optimasi yang berbeda, tim engineering dapat memusatkan upaya mereka pada Zenith sebagai lapisan performa standar, mengurangi biaya pelatihan dan operasional.

**3. Mendorong Inovasi Hardware yang Beragam:**
*   Dengan mengurangi friksi untuk mendukung *hardware* baru, Zenith dapat menciptakan pasar yang lebih kompetitif dan beragam untuk akselerator AI. Ini pada akhirnya dapat menurunkan harga dan meningkatkan pilihan untuk seluruh industri.

**Tabel 7.1: Ringkasan Nilai dan Dampak Zenith**
| Pemangku Kepentingan | Nilai Inti (Value Proposition) | Metrik Keberhasilan Kunci (KPI) |
| :--- | :--- | :--- |
| **Peneliti AI** | Akselerasi eksperimen, akses mudah ke berbagai hardware, reproduktibilitas. | Pengurangan waktu siklus pelatihan (target: 2x lebih cepat). Peningkatan jumlah eksperimen per kuartal. |
| **Insinyur ML/MLOps** | Penyederhanaan pipeline produksi, portabilitas hardware, jaminan performa. | Pengurangan waktu dari riset ke deployment (target: -50%). Pengurangan biaya inferensi cloud (target: -30%). |
| **Vendor Hardware** | Jalur adopsi yang lebih cepat untuk hardware baru. | Waktu untuk mendukung ekosistem ML (Time-to-Market) berkurang dari tahunan menjadi bulanan. |
| **Komunitas Open Source** | Konsolidasi upaya optimasi, peningkatan kualitas kode. | Jumlah kontributor lintas perusahaan. Peningkatan cakupan dan kekokohan pengujian di proyek upstream. |

Kesimpulannya, Zenith bukan sekadar alat teknis lainnya. Ia adalah **enabler strategis** yang dirancang untuk mengatasi beberapa inefisiensi struktural paling mendasar dalam ekosistem AI saat ini: fragmentasi, ketergantungan vendor (*vendor lock-in*), dan kompleksitas yang menghambat inovasi. Nilainya terletak pada kemampuannya untuk menyelaraskan insentif berbagai pemangku kepentingan di sekitar platform yang terbuka, dapat diverifikasi, dan berfokus pada performa. 

### **BAB 8: RENCANA JALAN (ROADMAP) DAN KESIMPULAN**

Bab penutup ini merangkum rencana konkret untuk mewujudkan Zenith, menyatakan kembali kesimpulan mendasar dari cetak biru, dan menguraikan langkah-langkah segera yang diperlukan untuk memulai perjalanan pengembangan yang ambisius ini.

#### **8.1. Roadmap Pengembangan Tahapan**
Roadmap Zenith dirancang dalam enam fase selama 36 bulan, dengan setiap fase menghasilkan artefak yang dapat diverifikasi dan memberikan landasan bagi fase berikutnya. Pendekatan ini meminimalkan risiko teknis dan memastikan bahwa setiap investasi pengembangan didasarkan pada fondasi yang kokoh.

**Tabel 8.1: Roadmap Pengembangan Zenith (Ringkasan 3 Tahun)**

| Fase | Periode | Fokus Utama | Milestone & Deliverable Kunci | Kriteria Kelulusan |
| :--- | :--- | :--- | :--- | :--- |
| **Fase 0: Persiapan** | Bulan 1-6 | Fondasi Teori & Desain | 1. Dokumen Spesifikasi Teknis Lengkap (TRD, SAD). <br> 2. Proof-of-Concept algoritma kunci (Python). <br> 3. Prototipe *adapter* ONNX & PyTorch sederhana. | Semua dokumen dan PoC disetujui melalui *peer review* internal. Hasil matematis dan alur konversi divalidasi. |
| **Fase 1: Inti & Abstraksi** | Bulan 7-12 | *Core Engine* & HAL Dasar | 1. *Core Engine* C++ dengan `GraphIR` dan *pass* optimasi dasar. <br> 2. Implementasi HAL: Backend CPU (SIMD) dan CUDA. <br> 3. *Binding* Python awal dengan PyBind11. | Engine dapat mengeksekusi model ONNX sederhana dengan benar dan lebih cepat dari *baseline* referensi di CPU/GPU. |
| **Fase 2: Kompatibilitas & Optimasi** | Bulan 13-18 | *Adapter* & *Backend* Tambahan | 1. *Adapter* produksi untuk PyTorch, TensorFlow, JAX. <br> 2. *Backend* ROCm dan oneAPI. <br> 3. *Pass* optimasi lanjutan (*fusion*, *auto-tuning* dasar). | ResNet-50/BERT dari ketiga *framework* dapat dikompilasi dan menunjukkan speedup >1.5x pada *hardware* target. |
| **Fase 3: Presisi & Kuantisasi** | Bulan 19-24 | Optimasi Agresif | 1. Manajemen presisi campuran (FP16/BF16) otomatis. <br> 2. Pipeline kuantisasi INT8 kalibrasi penuh. <br> 3. Sistem *fallback* dan validasi numerik otomatis. | Model FP32 dapat dikonversi ke INT8 dengan kehilangan akurasi < 0.5% (pada dataset validasi) dan tetap dieksekusi dengan benar. |
| **Fase 4: Pengujian & Pematangan** | Bulan 25-30 | Kualitas & Produksi | 1. *Suite* pengujian lengkap (unit, integrasi, performa, *fuzz*, *chaos*). <br> 2. Audit keamanan dan analisis kode statis. <br> 3. Dokumentasi komprehensif (pengguna, API, teknis). | *Code coverage* >95%, tidak ada *bug* kritis, *performance regression* <2%, dokumentasi lengkap. |
| **Fase 5: Rilis & Komunitas** | Bulan 31-36 | Peluncuran & Iterasi | 1. Rilis Alpha/Beta terbatas. <br> 2. Pembentukan repositori komunitas dan tata kelola. <br> 3. Rilis stabil pertama (v1.0.0). | Komunitas eksternal aktif memberikan umpan balik; v1.0.0 digunakan dalam setidaknya 3 proyek *proof-of-concept* eksternal. |

**Prinsip Pengelolaan Roadmap:**
1.  **Iteratif dan Berbasis Milestone**: Setiap fase bersifat *time-boxed*, tetapi kelulusannya bergantung pada pencapaian kriteria objektif, bukan hanya waktu. Jika milestone tidak tercapai, fase tersebut dapat diperpanjang sebelum melanjutkan.
2.  ***Feedback Loop* yang Ketat**: Hasil dari setiap fase (misalnya, benchmark Fase 1) akan digunakan untuk menyempurnakan prioritas dan desain fase berikutnya.
3.  **Keterbukaan dan Transparansi**: Ringkasan kemajuan dan temuan dari setiap fase akan didokumentasikan dan tersedia untuk umum (melalui *blog* teknis atau makalah) untuk membangun kepercayaan dan menarik kolaborasi sejak dini.

#### **8.2. Kesimpulan dan Peringatan Implementasi**
Zenith adalah cetak biru untuk sebuah **platform unifikasi dan optimasi yang model-agnostik dan hardware-agnostik** dalam ekosistem *Machine Learning*. Ia menjawab fragmentasi yang menghambat portabilitas, efisiensi, dan inovasi dengan menjadi lapisan "lem" cerdas yang mempertahankan kekuatan *framework* yang ada sembari membuka potensi penuh mereka di berbagai perangkat keras.

**Kesimpulan Utama:**
1.  **Visi yang Dibutuhkan**: Kebutuhan akan abstraksi yang lebih tinggi di atas *framework* dan *hardware* yang beragam adalah nyata dan mendesak, seperti yang dibuktikan oleh proliferasi tool serupa (ONNX Runtime, TVM, OpenXLA). Zenith membedakan diri dengan fokus pada **jaminan kinerja yang dapat diverifikasi secara matematis**, **integrasi yang sangat mudah (*zero-switch cost*)**, dan **komitmen pada universalitas perangkat keras yang sejati**.
2.  **Fondasi yang Kuat**: Cetak biru ini tidak dibangun di atas asumsi. Ia didasarkan pada **prinsip arsitektur perangkat lunak yang mapan** (lapisan abstraksi, kontrak API), **teori optimasi dan numerik yang solid**, dan pembelajaran dari **proyek *open source* dan industri yang sukses**.
3.  **Nilai Strategis Multi-Pihak**: Zenith memberikan nilai yang berbeda dan konkret bagi peneliti, insinyur, vendor *hardware*, dan komunitas *open source*, menciptakan peluang untuk kolaborasi yang saling menguntungkan dan mempercepat ekosistem secara keseluruhan.

**Peringatan dan Tantangan Kritis:**
1.  **Kompleksitas yang Luar Biasa**: Membangun *compiler* dan *runtime* yang benar-benar universal adalah salah satu tantangan rekayasa perangkat lunak yang paling kompleks. Risiko *scope creep* dan desain yang menjadi terlalu rumit sangat tinggi.
2.  **Sumber Daya dan Kepemimpinan**: Proyek dengan skala dan ambisi ini memerlukan **tim inti yang dedikatif dan sangat kompeten** (minimal 10-15 engineer senior bidang sistem, kompilator, dan ML) selama beberapa tahun, serta kepemimpinan teknis yang visioner dan kuat. Keterbatasan sumber daya manusia adalah risiko terbesar.
3.  **Adopsi dan Ekosistem**: Keberhasilan akhir Zenith bergantung pada adopsi. Membangun kemitraan awal dengan **satu atau dua *framework* utama** (misalnya, PyTorch) dan **beberapa vendor *hardware* kunci** (misalnya, NVIDIA dan Intel) sangat penting untuk mendapatkan daya tarik awal.
4.  **Kompetisi dan Koopetisi**: Ruang ini sudah memiliki pemain kuat (XLA, TVM, ONNX Runtime). Zenith harus menemukan ceruk uniknya dan/atau berkolaborasi dengan proyek-proyek tersebut, bukan mencoba menggantikan semuanya dari nol. Strategi "pembantu, bukan pengganti" adalah kunci.

#### **8.3. Langkah Selanjutnya**
Untuk memindahkan Zenith dari cetak biru menjadi kenyataan, tindakan berikut harus segera dimulai:

1.  **Pembentukan Tim Inti dan Penyusunan Proposal Detail**:
    *   Mengidentifikasi dan merekrut **Arsitek Utama** (dengan pengalaman dalam kompilator dan sistem ML) dan **Manajer Program Teknis**.
    *   Menyusun **Proposal Pengembangan Detail** berdasarkan cetak biru ini, termasuk estimasi anggaran, rencana rekrutmen, dan analisis risiko yang mendalam, untuk diajukan kepada calon sponsor atau investor.

2.  **Pengajuan dan Pembentukan Model Kolaborasi**:
    *   Menjajaki model **konsorsium industri** atau **proyek *open source* yang didanai bersama**, melibatkan universitas riset, perusahaan teknologi, dan vendor *hardware*. Model seperti ini dapat mendistribusikan biaya dan mengumpulkan keahlian yang diperlukan.
    *   Membuat **repositori GitHub organisasi** dan mendokumentasikan visi serta rencana awal untuk mulai membangun komunitas dan menerima masukan.

3.  **Inisiasi Proyek Percobaan (*Pilot Project*)**:
    *   Memulai implementasi **Fase 0** secara langsung dengan sumber daya kecil. Membuat *repository* privat untuk dokumen TRD/SAD dan kode PoC.
    *   Memfokuskan PoC pertama pada **alur kerja ONNX → Optimasi Grafik Sederhana → Eksekusi CPU**, untuk memvalidasi konsep inti dengan cepat dan menghasilkan artefak yang dapat ditunjukkan kepada calon mitra.

4.  **Outreach dan Membangun Kemitraan Awal**:
    *   Menjangkau penjaga (*maintainer*) proyek **PyTorch** (terutama tim `torch.compile`), **ONNX Runtime**, dan **Apache TVM** untuk berbagi visi dan menjajaki titik potensi integrasi atau kolaborasi.
    *   Menghubungi perwakilan dari **vendor *hardware*** (seperti Intel, AMD, ARM) yang mungkin memiliki minat strategis terhadap *hardware-agnostic framework*.

**Penutup**:  
Zenith bukan hanya sebuah proyek perangkat lunak; ia adalah sebuah proposal untuk masa depan yang lebih terbuka, efisien, dan kolaboratif dalam komputasi AI. Cetak biru ini memberikan peta jalan yang terperinci dan berdasarkan penelitian untuk mencapainya. Meskipun jalannya penuh dengan tantangan teknis dan organisasi yang berat, potensi imbalannya—mempercepat penemuan ilmiah, menghemat sumber daya komputasi yang besar, dan membuka pasar *hardware* yang lebih inovatif—sangatlah signifikan. Waktunya telah tiba untuk memulai perjalanan dari puncak konsep (*zenith of concept*) menuju realisasi yang berdampak.

---
**DAFTAR PUSTAKA**  
*(Catatan: Bagian ini akan berisi 50+ referensi ke makalah akademis, dokumentasi proyek *open source*, artikel teknis, dan situs web yang dikutip atau dijadikan inspirasi sepanjang dokumen. Contoh struktur:)*

1.  **PyTorch**. "PyTorch 2.0: `torch.compile`". https://pytorch.org/get-started/pytorch-2.0/.
2.  **TensorFlow**. "XLA: Optimizing Compiler for Machine Learning". https://www.tensorflow.org/xla.
3.  **Google Research**. "JAX: Composable Transformations of Python+NumPy Programs". https://github.com/google/jax.
4.  ONNX Steering Committee. "Open Neural Network Exchange (ONNX) Specification". https://github.com/onnx/onnx.
5.  Lattner, C., et al. "MLIR: A Compiler Infrastructure for the End of Moore's Law". arXiv:2002.11054 [cs.PL]. 2020.
6.  Chen, T., et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning". In *OSDI*. 2018.
7.  NVIDIA. "CUDA Deep Neural Network library (cuDNN)". https://developer.nvidia.com/cudnn.
8.  Intel. "oneAPI Deep Neural Network Library (oneDNN)". https://github.com/oneapi-src/oneDNN.
9.  AMD. "ROCm: Open-Source Platform for HPC and Ultrascale Computing". https://www.amd.com/en/graphics/servers-solutions-rocm.
10. ... (dan seterusnya hingga lebih dari 50 entri).

**LAMPIRAN**  
*(Berisi diagram teknis tambahan, contoh kode yang lebih panjang, matriks fitur yang detail, atau glosarium istilah.)*

---
**DOKUMEN INI DISUSUN OLEH:**  
WAHYU ARDIANSYAH  
*Arsitek Utama, Proposal Zenith*  
Tanggal Penyelesaian: 16 Desember 2025  
Status: Cetak Biru Versi 1.0 – Untuk Review dan Aksi

**© 2025 Zenith Project Proposal. Dokumen ini dapat didistribusikan secara bebas untuk tujuan non-komersial dan review, dengan atribusi kepada penulis.**