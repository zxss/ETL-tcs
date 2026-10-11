# -*- coding: utf-8 -*-
"""Карусель для LinkedIn: 12 страниц 1080×1350 (4:5)."""
import io

CSS = """
@page { size: 11.25in 14.0625in; margin: 0; }
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#0E1520;-webkit-print-color-adjust:exact;print-color-adjust:exact}
:root{
  --ink:#0E1520; --ink-2:#161F2C; --line:#27323F;
  --fg:#EEF3F8; --fg-2:#A9B7C6; --fg-3:#6E7F91;
  --acc:#6FB6DB; --acc-d:#1D4E6F;
  --fact:#4FBE92; --hypo:#DCA748; --warn:#E07A5F;
  --serif:"IBM Plex Serif",Georgia,serif;
  --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",Menlo,monospace;
}
.slide{width:1080px;height:1350px;background:var(--ink);color:var(--fg);
  font-family:var(--sans);padding:74px 76px 64px;display:flex;flex-direction:column;
  page-break-after:always;break-after:page;position:relative;overflow:hidden}
.slide:last-child{page-break-after:auto}
.top{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--mono);font-size:19px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--fg-3);padding-bottom:22px;border-bottom:1px solid var(--line);flex:0 0 auto}
.top .acc{color:var(--acc)}
.mid{flex:1 1 auto;display:flex;flex-direction:column;justify-content:center;padding-block:44px;min-height:0}
.bot{flex:0 0 auto;border-top:1px solid var(--line);padding-top:22px;
  font-family:var(--mono);font-size:18px;line-height:1.5;color:var(--fg-3)}
h1{font-family:var(--serif);font-weight:700;font-size:96px;line-height:1.04;letter-spacing:-.02em}
h2{font-family:var(--serif);font-weight:600;font-size:60px;line-height:1.12;letter-spacing:-.015em;margin-bottom:40px}
h2 em{font-style:normal;color:var(--acc)}
p{font-size:32px;line-height:1.48;color:var(--fg-2)}
p+p{margin-top:26px}
p strong{color:var(--fg);font-weight:600}
.kicker{font-family:var(--mono);font-size:22px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);margin-bottom:34px}
.lede{font-size:38px;line-height:1.38;color:var(--fg)}
.huge{font-family:var(--sans);font-weight:600;font-size:148px;line-height:1.02;letter-spacing:-.035em;color:var(--acc);font-variant-numeric:tabular-nums}
.huge.warn{color:var(--warn)} .huge.ok{color:var(--fact)}
.sub{font-size:30px;line-height:1.45;color:var(--fg-2);margin-top:26px}
.rows{display:flex;flex-direction:column;gap:0;margin-top:14px}
.row{display:flex;gap:30px;align-items:baseline;padding:26px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row .v{font-family:var(--sans);font-weight:600;font-size:48px;color:var(--acc);
  flex:0 0 300px;text-align:right;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.row .v.sm{font-size:38px;font-family:var(--mono);font-weight:600}
.row .k{font-size:29px;line-height:1.4;color:var(--fg-2)}
.row .k b{color:var(--fg);font-weight:600}
.cols{display:flex;gap:28px;margin-top:10px}
.col{flex:1;background:var(--ink-2);border:1px solid var(--line);border-radius:6px;padding:32px 30px}
.col h3{font-family:var(--mono);font-size:20px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--fg-3);font-weight:500;margin-bottom:20px}
.col .big{font-family:var(--mono);font-weight:600;font-size:60px;line-height:1.05;color:var(--acc);margin-bottom:16px;letter-spacing:-.03em}
.col p{font-size:26px;line-height:1.4}
ul{list-style:none;margin-top:8px}
li{font-size:30px;line-height:1.42;color:var(--fg-2);padding:20px 0 20px 46px;
  border-bottom:1px solid var(--line);position:relative}
li:last-child{border-bottom:0}
li::before{content:"";position:absolute;left:8px;top:32px;width:16px;height:16px;
  border:2px solid var(--fg-3);border-radius:3px}
li b{color:var(--fg);font-weight:600}
.mk{font-family:var(--mono);font-size:.74em;font-weight:500;padding:3px 9px;border-radius:3px;
  vertical-align:.22em;letter-spacing:.03em;white-space:nowrap}
.mk-f{color:var(--fact);background:rgba(79,190,146,.14)}
.mk-g{color:var(--hypo);background:rgba(220,167,72,.14)}
.tag{display:inline-block;font-family:var(--mono);font-size:20px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--acc);border:1px solid var(--acc-d);
  border-radius:4px;padding:8px 14px;margin-right:12px;margin-bottom:12px}
.rule{height:1px;background:var(--line);margin:34px 0}
.swipe{position:absolute;right:76px;bottom:64px;font-family:var(--mono);font-size:19px;
  letter-spacing:.14em;text-transform:uppercase;color:var(--acc)}
"""

# (заголовок сверху-слева, содержимое, подпись снизу)
S = []

S.append(("", """
<div class="mid">
  <p class="kicker">Стратегическое исследование · сентябрь 2026</p>
  <h1>Алгоритмические<br/>торговые системы</h1>
  <p class="lede" style="margin-top:46px;max-width:27ch">Гонка за скоростью закончилась. Началась гонка за дисциплину доказательства.</p>
</div>
<div class="bot">4 исследовательских стрима · ≈60 источников · 18 000 слов<span class="swipe">Листайте →</span></div>"""))

S.append(("Метод", """
<div class="mid">
  <h2>Каждое утверждение здесь несёт <em>маркер достоверности</em></h2>
  <p>В теме, где реальные задержки, ценники на данные и экономика фондов раскрываются выборочно, отчёт без такого разделения читался бы солиднее — но был бы хуже как основание для решения.</p>
  <div class="rule"></div>
  <div class="rows">
    <div class="row"><span class="v sm" style="color:var(--fact)">Ф</span><span class="k"><b>Факт</b> — подтверждён открытым источником</span></div>
    <div class="row"><span class="v sm">В</span><span class="k"><b>Вывод</b> — интерпретация, логика показана явно</span></div>
    <div class="row"><span class="v sm" style="color:var(--hypo)">Г</span><span class="k"><b>Гипотеза</b> — оценка порядка величины без подтверждения</span></div>
  </div>
</div>
<div class="bot">32 позиции, подтвердить которые не удалось, вынесены в отдельное приложение</div>"""))

S.append(("Главный вывод", """
<div class="mid">
  <h2>Барьер входа вырос <em>и</em> упал — в разных сегментах одновременно</h2>
  <div class="cols">
    <div class="col"><h3>Вырос</h3><div class="big">3–4<br/>порядка</div>
      <p>разрыв tick-to-trade между обычным программным стеком и FPGA-фронтиром. Это резкая граница, а не градиент.</p></div>
    <div class="col"><h3>Упал</h3><div class="big">×100</div>
      <p>падение стоимости входа в средние горизонты: нормализованные тиковые данные — от 199 $/мес, production-движок — открытый код.</p></div>
  </div>
  <p style="margin-top:40px"><strong>Дефицитный ресурс сместился с инфраструктуры на способность отличить реальную закономерность от переобученного шума.</strong></p>
</div>
<div class="bot">Раздел 1. Executive Summary</div>"""))

S.append(("Сегмент скорости", """
<div class="mid">
  <h2>В скоростном сегменте конкуренция уже <em>закрыта</em></h2>
  <div class="rows">
    <div class="row"><span class="v">39,6</span><span class="k"><b>Jane Street</b>, млрд $ чистой торговой выручки за 2025 — при ≈3 000 сотрудников <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v">12,3</span><span class="k"><b>Hudson River Trading</b>, млрд $ <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v">12,2</span><span class="k"><b>Citadel Securities</b>, млрд $, +25 % <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v">30 %</span><span class="k">доля Jane Street и Citadel Securities в глобальном пуле торговой выручки 114 млрд $ <span class="mk mk-f">Ф</span></span></div>
  </div>
  <p style="margin-top:36px">Новый участник конкурирует не с департаментом банка, а с частной компанией, где почти вся прибыль уходит в R&amp;D и железо.</p>
</div>
<div class="bot">Раздел 2. Ландшафт</div>"""))

S.append(("Осторожно", """
<div class="mid">
  <p class="kicker">Но рекорд 2025 — это волатильность, а не маржа</p>
  <h2>Выручка маркетмейкера — <em>функция волатильности</em>, а не стабильный спред</h2>
  <div class="rows">
    <div class="row"><span class="v sm">7,2 → &lt;1</span><span class="k">млрд $ — выручка HFT в акциях США: пик 2009 → ниже 1 млрд к 2017 <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v sm">31 %</span><span class="k">разброс среднедневного дохода Virtu <b>внутри одного 2025 года</b> при неизменных стратегиях <span class="mk mk-f">Ф</span></span></div>
  </div>
  <p style="margin-top:36px"><strong>Следствие:</strong> доходность этого класса стратегий скоррелирована по всей индустрии — драйвер один. Диверсификация внутри класса не работает.</p>
</div>
<div class="bot">Раздел 2. Ландшафт</div>"""))

S.append(("Альфа", """
<div class="mid">
  <h2>Предсказуемость есть. <em>Торгуемости нет.</em></h2>
  <p style="margin-bottom:30px">Три независимых измерения одного и того же:</p>
  <div class="rows">
    <div class="row"><span class="v sm">TLOB</span><span class="k">заявляет SoTA на книге заявок — и тут же показывает, что при пороге, равном <b>среднему спреду</b>, качество существенно деградирует <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v sm">DeepLOB</span><span class="k">прибыль есть только при исполнении <b>по середине спреда</b> — без спреда, комиссии и очереди <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v sm">RL</span><span class="k">выигрыш над TWAP — <b>0,3–1,5 б.п.</b> против 8 б.п. типичной комиссии на круг <span class="mk mk-f">Ф</span></span></div>
  </div>
</div>
<div class="bot">Раздел 3. Генерация альфы</div>"""))

S.append(("LLM", """
<div class="mid">
  <p class="kicker">Утечка данных перестала быть возражением — её измерили</p>
  <div class="huge warn">−51…−62 %</div>
  <p class="sub">деградация коэффициента Шарпа торговых агентов на LLM при переносе тестового окна <strong>за пределы обучающего корпуса модели</strong>. Полная доходность падает на 50–72 %. <span class="mk mk-f">Ф</span></p>
  <div class="rule"></div>
  <p><strong>82 %</strong> предсказаний одного из агентов не меняются при значимом возмущении входных данных — он воспроизводит запомненное, а не реагирует на рынок. <span class="mk mk-f">Ф</span></p>
</div>
<div class="bot">Profit Mirage, arXiv:2510.07920 · 10 протестированных систем</div>"""))

S.append(("Потолок ML", """
<div class="mid">
  <h2>Потолок честного ML известен, и он <em>скромный</em></h2>
  <div class="cols">
    <div class="col"><h3>Месячный R² вне выборки</h3><div class="big">0,4 %</div>
      <p>на уровне отдельной акции. Всё, что заявляет на порядок больше на сопоставимой задаче, требует проверки на утечку. <span class="mk mk-f">Ф</span></p></div>
    <div class="col"><h3>Шарп long-short портфеля</h3><div class="big">1,35</div>
      <p>из того же R². Портфельная агрегация усиливает микроскопический сигнал: <b>охват важнее точности</b>. <span class="mk mk-f">Ф</span></p></div>
  </div>
  <p style="margin-top:38px">Обычный многослойный перцептрон обходит state-of-the-art на бенчмарке книги заявок. Сложность архитектуры не окупается, пока не исчерпаны простые модели. <span class="mk mk-f">Ф</span></p>
</div>
<div class="bot">Gu, Kelly, Xiu · Review of Financial Studies, 2020</div>"""))

S.append(("Ёмкость", """
<div class="mid">
  <h2>Ёмкость растёт как <em>квадрат</em> преимущества</h2>
  <p>По закону квадратного корня рыночного воздействия — подтверждён на акциях, фьючерсах, опционах и биткойне: <span class="mk mk-f">Ф</span></p>
  <div class="rows" style="margin-top:22px">
    <div class="row"><span class="v sm">÷2</span><span class="k">уменьшили edge вдвое — <b>ёмкость упала вчетверо</b></span></div>
    <div class="row"><span class="v sm">2/3</span><span class="k">в оптимуме <b>две трети валовой альфы</b> уходит в собственное воздействие на цену</span></div>
  </div>
  <div class="rule"></div>
  <p><strong>Шарп и ёмкость обратно связаны.</strong> Стратегии с Шарпом 5 упираются в 10–100 млн $ работающего капитала; трендследование с Шарпом 0,7 масштабируется на десятки миллиардов.</p>
  <p style="margin-top:22px">Порядок оценки: <strong>сначала ёмкость, потом TCO, и только потом Шарп.</strong></p>
</div>
<div class="bot">Раздел 6. Экономика</div>"""))

S.append(("Экономика", """
<div class="mid">
  <p class="kicker">Точка безубыточности при расходах 1,5 млн $ в год</p>
  <div class="huge">55,4 млн $</div>
  <p class="sub">под управлением нужно фонду с коэффициентом Шарпа <strong>1,0</strong>, чтобы выйти в ноль. При Шарпе 2,0 — 34,7 млн $. <span class="mk mk-g">Г, расчёт</span></p>
  <div class="rule"></div>
  <p>Фонд с <strong>25 млн $</strong> и Шарпом 1,0 зарабатывает 678 тыс. $ при расходах 1,5 млн — теряет <strong>0,8 млн $ в год</strong>. Фонд в 10–20 млн структурно убыточен.</p>
  <p style="margin-top:22px">Розничный алготрейдинг с капиталом менее <strong>0,6–1,2 млн $</strong> — покупка опциона на обучение, а не бизнес.</p>
</div>
<div class="bot">Раздел 6. Плейбук «покупать или строить»</div>"""))

S.append(("Base rate", """
<div class="mid">
  <h2>Априорная вероятность успеха — <em>1–15 %</em></h2>
  <div class="rows">
    <div class="row"><span class="v">97 %</span><span class="k">дей-трейдеров Бразилии, продержавшихся более 300 дней, <b>потеряли деньги</b>. Выше минимальной зарплаты заработали 1,1 % <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v">84 %</span><span class="k">из ~360 тыс. дей-трейдеров Тайваня убыточны; предсказуемо прибыльны около 1 % <span class="mk mk-f">Ф</span></span></div>
    <div class="row"><span class="v">26 %</span><span class="k">CTA-фондов выживают 5 лет — при годовой аттриции 23,5 % <span class="mk mk-f">Ф</span></span></div>
  </div>
  <p style="margin-top:34px"><strong>Механизм, а не только результат:</strong> тайваньский трейдер теряет 7 б.п. в день до издержек — издержки увеличивают убыток более чем втрое, до 23,9 б.п. <span class="mk mk-f">Ф</span></p>
</div>
<div class="bot">Полные регуляторные данные, а не самоотбор выживших</div>"""))

S.append(("Что делать", """
<div class="mid">
  <h2>Чек-лист инвестиционного комитета</h2>
  <p style="margin-bottom:20px">Применять к любому питчу стратегии — внутреннему и внешнему:</p>
  <ul>
    <li>Сколько <b>независимых</b> конфигураций испытано? Нет ответа — нет результата</li>
    <li>Показан <b>дефлированный</b> Шарп, а не сырой?</li>
    <li>Вероятность переобучения бэктеста ниже 50 %?</li>
    <li>Модель издержек написана <b>до</b> стратегии?</li>
    <li>Оценена <b>ёмкость</b>, а не только доходность?</li>
    <li>Для LLM: тестовое окно строго <b>позже</b> knowledge cutoff?</li>
  </ul>
  <p style="margin-top:30px">При 316 протестированных факторах и нулевой истинной альфе на уровне 5 % ожидается <strong>≈16 «значимых» находок чисто случайно</strong>. <span class="mk mk-f">Ф</span></p>
</div>
<div class="bot">Полный отчёт — 18 000 слов, 7 разделов — ссылка в первом комментарии</div>"""))

parts = []
for idx, (label, inner) in enumerate(S, start=1):
    if idx == 1:
        top = ('<div class="top"><span class="acc">Whitepaper</span>'
               '<span>%02d / %d</span></div>' % (idx, len(S)))
    else:
        top = ('<div class="top"><span>%s</span><span class="acc">%02d / %d</span></div>'
               % (label, idx, len(S)))
    parts.append('<section class="slide">' + top + inner + "</section>")

page = ("<title>Карусель LinkedIn</title><style>" + CSS + "</style>" + "".join(parts))
io.open("carousel.html", "w", encoding="utf-8").write(page)
print("страниц:", len(S), "| байт:", len(page.encode("utf-8")))
