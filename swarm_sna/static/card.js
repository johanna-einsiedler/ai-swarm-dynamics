/* The report card's figures. D3 v7; one renderer per figure kind, each reading the JSON
   payload the Python side embeds. Group filtering (era) re-renders the figures that
   declare data-groups; everything else renders once and on resize. */
(function () {
  "use strict";
  document.querySelectorAll(".nojs").forEach(e => e.remove());  // the script runs, so the warning for viewers that block it goes
  const D = JSON.parse(document.getElementById("card-data").textContent);
  const P = D.palette;
  const state = { group: "all" };
  const f3 = d3.format(".3f"), f2 = d3.format(".2f"), pct = d3.format(".0%"), int = d3.format(",");
  const pfmt = p => (p == null || isNaN(p)) ? "n/a" : p < 0.002 ? "p < 0.002" : "p = " + f3(p);
  const familyColour = f => { const i = D.family_order.indexOf(f); return i >= 0 && i < P.series.length ? P.series[i] : P.other; };
  const thin = new Set(D.groups.filter(g => g.thin).map(g => g.name));
  const G = new Map(D.groups.map(g => [g.name, g]));
  const glabel = name => { const g = G.get(name); return g && g.marker ? g.marker + " " + name : name; };  // "▲ era 2"
  const groupsShown = () => state.group === "all" ? D.groups.map(g => g.name) : [state.group];
  const width = (el, min) => Math.max(min || 320, el.clientWidth || 800);

  // ---- one tooltip for the page; names come from data, so textContent only
  const tip = d3.select("body").append("div").attr("class", "tip").style("display", "none");
  function showTip(event, title, rows) {
    tip.style("display", "block").selectAll("*").remove();
    if (title) tip.append("div").attr("class", "tip-title").text(title);
    (rows || []).forEach(([label, value, colour]) => {
      const r = tip.append("div").attr("class", "tip-row");
      if (colour) r.append("span").attr("class", "key").style("background", colour);
      r.append("b").text(value);
      r.append("span").text(label);
    });
    moveTip(event);
  }
  function moveTip(event) {
    const w = tip.node().offsetWidth, h = tip.node().offsetHeight;
    const right = event.pageX + 16 + w > window.scrollX + document.documentElement.clientWidth - 8;
    tip.style("left", (right ? event.pageX - w - 12 : event.pageX + 16) + "px").style("top", Math.max(4, event.pageY - h / 2) + "px");
  }
  const hideTip = () => tip.style("display", "none");

  const svgIn = (el, w, h) => d3.select(el).append("svg").attr("class", "chart").attr("viewBox", [0, 0, w, h]).attr("width", w).attr("height", h);
  function axis(g) {
    g.selectAll(".domain").attr("stroke", P.axis);
    g.selectAll(".tick line").attr("stroke", P.axis);
    g.selectAll("text").attr("fill", P.muted).attr("font-size", 10.5);
    return g;
  }
  function grid(g, scale, ticks, length, horizontal) {
    g.selectAll("line").data(scale.ticks(ticks)).join("line")
      .attr(horizontal ? "y1" : "x1", d => scale(d)).attr(horizontal ? "y2" : "x2", d => scale(d))
      .attr(horizontal ? "x1" : "y1", 0).attr(horizontal ? "x2" : "y2", length).attr("stroke", P.grid);
  }
  function legend(el, items, line) {
    const l = d3.select(el).append("div").attr("class", "legend");
    items.forEach(([label, colour]) => { const s = l.append("span"); s.append("i").attr("class", line ? "line" : null).style("background", colour); s.append("span").text(label); });
  }
  const familyLegend = el => legend(el, D.family_order.map(f => [f, familyColour(f)]).concat([["other", P.other]]));
  const note = (el, text) => d3.select(el).append("p").attr("class", "note").text(text);
  function wrap(text, maxChars) {  // two lines at most
    if (text.length <= maxChars) return [text];
    const i = text.lastIndexOf(" ", maxChars);
    return i > 0 ? [text.slice(0, i), text.slice(i + 1)] : [text];
  }
  function endBar(g, x0, x1, y, h) {  // a bar square at the baseline and rounded at the data end
    const w = Math.max(0, x1 - x0), r = Math.min(4, w / 2);
    return g.append("path").attr("d", `M${x0},${y} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${h - 2 * r} a${r},${r} 0 0 1 -${r},${r} h-${w - r} z`);
  }

  // ---- null distributions: one cell per statistic x group, histogram when the draws were saved
  function nulls(el) {
    const N = D.nulls; if (!N) return;
    const groups = groupsShown().filter(g => N.table.some(r => r.group === g));
    if (!groups.length) return;
    const kinds = Object.keys(N.null_names);
    const kind = el._null || (kinds.includes(D.default_null) ? D.default_null : kinds[0]);
    if (kinds.length > 1) {
      const ctl = d3.select(el).append("div").attr("class", "ctl");
      ctl.append("span").attr("class", "ctl-label").text("null");
      ctl.selectAll("button").data(kinds).join("button").attr("class", k => "seg" + (k === kind ? " on" : "")).text(k => N.null_names[k])
        .on("click", (e, k) => { el._null = k; el.replaceChildren(); nulls(el); });
    }
    const W = width(el), left = 112, top = 22, rowH = 104;
    const colW = Math.max(150, (W - left) / groups.length);
    const svg = svgIn(el, W, top + rowH * N.stats.length);
    N.stats.forEach((s, r) => {
      const y0 = top + r * rowH;
      wrap(s.label, 16).forEach((line, i) => svg.append("text").attr("x", 0).attr("y", y0 + 30 + i * 13).attr("fill", P.ink).attr("font-size", 11.5).attr("font-weight", 600).text(line));
      groups.forEach((g, c) => {
        const row = N.table.find(t => t.statistic === s.key && t.group === g && t.null === kind);
        if (!row) return;
        const x0 = left + c * colW, w = colW - 22, h = rowH - 32;
        const draws = N.draws && N.draws[kind] && N.draws[kind][s.key] && N.draws[kind][s.key][g];
        const cell = svg.append("g").attr("transform", `translate(${x0},${y0})`);
        if (r === 0) cell.append("text").attr("y", -8).attr("fill", thin.has(g) ? P.muted : P.ink).attr("font-size", 11.5).attr("font-weight", 600).text(glabel(g) + (thin.has(g) ? "  (thin)" : ""));
        const lo = Math.min(row.null_lo, row.observed, draws ? d3.min(draws) : Infinity), hi = Math.max(row.null_hi, row.observed, draws ? d3.max(draws) : -Infinity);
        const pad = (hi - lo) * 0.1 || Math.max(Math.abs(hi) * 0.02, 0.005);  // a degenerate null (every draw equal) still gets a readable axis
        const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);
        if (draws && draws.length) {
          const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(28))(draws);
          const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).range([h, h * 0.5]);
          cell.selectAll("rect.bin").data(bins).join("rect").attr("class", "bin").attr("x", b => x(b.x0) + 0.5).attr("width", b => Math.max(0.5, x(b.x1) - x(b.x0) - 1))
            .attr("y", b => y(b.length)).attr("height", b => h - y(b.length)).attr("fill", P.null);
        } else {
          cell.append("rect").attr("x", x(row.null_lo)).attr("width", Math.max(2, x(row.null_hi) - x(row.null_lo))).attr("y", h * 0.65).attr("height", 10).attr("rx", 2).attr("fill", P.null);
          cell.append("circle").attr("cx", x(row.null_mean)).attr("cy", h * 0.65 + 5).attr("r", 3.5).attr("fill", P.muted).attr("stroke", "#fff").attr("stroke-width", 2);
        }
        cell.append("line").attr("x1", x(row.observed)).attr("x2", x(row.observed)).attr("y1", 2).attr("y2", h).attr("stroke", P.observed).attr("stroke-width", 2);
        cell.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(4).tickSize(3)).call(axis);
        const start = row.observed > row.null_mean, tx = start ? 2 : w - 2;
        const t = cell.append("text").attr("x", tx).attr("y", 10).attr("text-anchor", start ? "start" : "end").attr("font-size", 10).attr("fill", P.ink);
        t.append("tspan").attr("x", tx).text("observed " + f3(row.observed));
        t.append("tspan").attr("x", tx).attr("dy", 11).attr("fill", P.muted).text("null " + f3(row.null_mean));
        t.append("tspan").attr("x", tx).attr("dy", 11).attr("fill", P.muted).text(pfmt(row.p));
        cell.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent")
          .on("pointermove", e => showTip(e, s.label + " · " + g, [["observed", f3(row.observed), P.observed], ["null mean", f3(row.null_mean), P.null], ["null 95% band", `[${f3(row.null_lo)}, ${f3(row.null_hi)}]`], [N.null_names[kind], pfmt(row.p)]]))
          .on("pointerleave", hideTip);
      });
    });
    note(el, (N.draws ? "Grey: the statistic on each permuted event stream. " : "Grey: the null's 95% band, dot its mean. ") + "Orange: the observed value.");
  }

  // ---- mention network per group, force layout
  function network(el) {
    const groups = groupsShown().filter(g => D.networks[g]);
    if (!groups.length) return;
    const W = width(el), n = groups.length;
    const cols = n === 1 ? 1 : Math.min(n, W >= 860 ? 3 : W >= 560 ? 2 : 1), pw = W / cols;
    const ph = n === 1 ? Math.min(540, pw * 0.7) : Math.min(400, pw * 0.95);
    const svg = svgIn(el, W, ph * Math.ceil(n / cols));
    groups.forEach((g, i) => {
      const net = D.networks[g];
      const panel = svg.append("g").attr("transform", `translate(${(i % cols) * pw},${Math.floor(i / cols) * ph})`);
      drawForce(panel, net.nodes, net.edges.map(e => ({ source: e.s, target: e.t, w: e.w })), pw, ph, `${glabel(g)}  (${net.nodes.length} agents)` + (thin.has(g) ? " · thin" : ""), false,
        d => [["mentions received", int(Math.round(d.in))], ["mentions sent", int(Math.round(d.out))], ["messages", int(d.msgs)]], d => d.in);
    });
    familyLegend(el);
    note(el, "Node size: mentions received. Strongest ties only. Hover an agent to see its ties.");
  }

  function drawForce(g, nodeData, linkData, w, h, title, arrows, tipRows, size, edgeTip) {
    const all = nodeData.map(d => Object.assign({}, d)), byId = new Map(all.map(d => [d.id, d]));
    // the strongest ties only: about three per agent; an agent with none of them is counted, not drawn
    const links = linkData.filter(l => byId.has(l.source) && byId.has(l.target)).sort((a, b) => b.w - a.w).slice(0, Math.max(24, 3 * all.length)).map(l => Object.assign({}, l));
    const tied = new Set(links.flatMap(l => [l.source, l.target]));
    const nodes = all.filter(d => tied.has(d.id)), isolated = all.length - nodes.length;
    const subtitle = isolated ? `${isolated} agent${isolated > 1 ? "s" : ""} without a tie among the strongest not drawn` : "";
    const wmax = d3.max(links, l => l.w) || 1, smax = d3.max(nodes, size) || 1;
    const r = d => 4 + 15 * Math.sqrt(size(d) / smax);
    const k = Math.min(w, h), n = Math.max(1, nodes.length);  // forces scale with the panel, so a few agents still fill it, and ease off as agents multiply
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(l => k * (0.14 + 0.3 * (1 - l.w / wmax))).strength(l => 0.08 + 0.3 * l.w / wmax))
      .force("charge", d3.forceManyBody().strength(-1.6 * k * Math.min(1, 12 / n)))
      .force("x", d3.forceX(w / 2).strength(0.1)).force("y", d3.forceY(h / 2 + 8).strength(0.1))  // gravity keeps the loosely tied in the picture
      .force("collide", d3.forceCollide(d => r(d) + 6).iterations(2)).stop();
    for (let i = 0; i < 400; i++) sim.tick();
    // Fit the layout to the panel by scaling and centring it, never by clamping: clamping piles the outer agents onto the edge.
    const xs = d3.extent(nodes, d => d.x), ys = d3.extent(nodes, d => d.y), padX = 52, padTop = 50, padBottom = 16;
    const s = Math.min((w - 2 * padX) / Math.max(1, xs[1] - xs[0]), (h - padTop - padBottom) / Math.max(1, ys[1] - ys[0]), 1.5);
    const offX = padX + ((w - 2 * padX) - (xs[1] - xs[0]) * s) / 2, offY = padTop + ((h - padTop - padBottom) - (ys[1] - ys[0]) * s) / 2;
    nodes.forEach(d => { d.x = offX + (d.x - xs[0]) * s; d.y = offY + (d.y - ys[0]) * s; });
    g.append("text").attr("x", 8).attr("y", 16).attr("fill", P.ink).attr("font-size", 12).attr("font-weight", 600).text(title);
    if (subtitle) g.append("text").attr("x", 8).attr("y", 30).attr("fill", P.muted).attr("font-size", 10.5).text(subtitle);
    if (arrows) g.append("defs").append("marker").attr("id", "arrow").attr("viewBox", "0 -4 8 8").attr("refX", 8).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
      .append("path").attr("d", "M0,-3.5L8,0L0,3.5").attr("fill", P.muted);
    const end = l => { // shorten to the target's rim so an arrowhead is visible
      const dx = l.target.x - l.source.x, dy = l.target.y - l.source.y, len = Math.hypot(dx, dy) || 1, k = (len - r(l.target) - 3) / len;
      return [l.source.x + dx * k, l.source.y + dy * k];
    };
    const edge = g.append("g").selectAll("line").data(links).join("line")
      .attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => arrows ? end(l)[0] : l.target.x).attr("y2", l => arrows ? end(l)[1] : l.target.y)
      .attr("stroke", arrows ? P.muted : P.faint).attr("stroke-opacity", arrows ? 0.55 : 1).attr("stroke-width", l => 0.6 + 3 * l.w / wmax).attr("stroke-linecap", "round").attr("marker-end", arrows ? "url(#arrow)" : null);
    const node = g.append("g").selectAll("circle").data(nodes).join("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", r)
      .attr("fill", d => familyColour(d.family)).attr("stroke", "#fff").attr("stroke-width", 2);
    const labelled = new Set(nodes.slice().sort((a, b) => size(b) - size(a)).slice(0, nodes.length <= 14 ? 14 : 6).map(d => d.id));
    g.append("g").selectAll("text").data(nodes.filter(d => labelled.has(d.id))).join("text").attr("x", d => Math.max(50, Math.min(w - 50, d.x))).attr("y", d => d.y - r(d) - 4)
      .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", P.ink).text(d => d.id.length > 22 ? d.id.slice(0, 21) + "…" : d.id);
    if (edgeTip) g.append("g").selectAll("line").data(links).join("line")
      .attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => l.target.x).attr("y2", l => l.target.y).attr("stroke", "transparent").attr("stroke-width", 10)
      .on("pointermove", (e, l) => showTip(e, `${l.source.id} → ${l.target.id}`, edgeTip(l))).on("pointerleave", hideTip);
    const touches = (l, d) => l.source === d || l.target === d;
    g.append("g").selectAll("circle").data(nodes).join("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", d => Math.max(12, r(d) + 4)).attr("fill", "transparent")
      .on("pointerenter", (e, d) => {
        edge.attr("stroke", l => touches(l, d) ? P.ink : (arrows ? P.muted : P.faint)).attr("stroke-opacity", l => touches(l, d) ? 0.8 : 0.25);
        node.attr("opacity", o => o === d || links.some(l => touches(l, d) && touches(l, o)) ? 1 : 0.3);
      })
      .on("pointermove", (e, d) => showTip(e, d.id + " · " + d.family, tipRows(d)))
      .on("pointerleave", () => { edge.attr("stroke", arrows ? P.muted : P.faint).attr("stroke-opacity", arrows ? 0.55 : 1); node.attr("opacity", 1); hideTip(); });
  }

  // ---- the network over time: agents on a circle in join order, a slider over weeks
  const T = { i: null, mode: "mentions", win: 4, timer: null };
  function timeline(el) {
    const L = D.timeline; if (!L || !L.periods.length) return;
    const periods = L.periods, n = L.agents.length;
    if (T.i === null) T.i = periods.length - 1;
    const modes = [["mentions", "who mentions whom"]].concat(L.help ? [["help", "who answers whose requests"]] : []);
    if (!modes.some(m => m[0] === T.mode)) T.mode = "mentions";
    const rerender = () => { el.replaceChildren(); timeline(el); };
    const ctl = d3.select(el).append("div").attr("class", "ctl");
    ctl.append("span").attr("class", "ctl-label").text("network");
    ctl.selectAll("button.mode").data(modes).join("button").attr("class", m => "seg mode" + (m[0] === T.mode ? " on" : "")).text(m => m[1]).on("click", (e, m) => { T.mode = m[0]; rerender(); });
    ctl.append("span").attr("class", "ctl-label").text("window");
    ctl.selectAll("button.win").data([1, 4, 13]).join("button").attr("class", w => "seg win" + (w === T.win ? " on" : "")).text(w => w === 1 ? "1 week" : w + " weeks").on("click", (e, w) => { T.win = w; rerender(); });
    const play = ctl.append("button").attr("class", "seg").text(T.timer ? "pause" : "play").on("click", () => {
      if (T.timer) { clearInterval(T.timer); T.timer = null; play.text("play"); return; }
      if (T.i >= periods.length - 1) T.i = 0;
      play.text("pause");
      T.timer = setInterval(() => { if (T.i >= periods.length - 1) { clearInterval(T.timer); T.timer = null; play.text("play"); return; } T.i += 1; slider.property("value", T.i); draw(); }, 650);
    });
    const stamp = ctl.append("span").attr("class", "stamp");
    const slider = d3.select(el).append("input").attr("type", "range").attr("class", "slider").attr("min", 0).attr("max", periods.length - 1).attr("value", T.i)
      .on("input", function () { T.i = +this.value; draw(); });

    // the strip: total tie weight per week, era boundaries, the current window marked
    const W = width(el), sh = 54;
    const totals = periods.map(p => d3.sum(L[T.mode][p] || [], e => e[2]));
    const strip = svgIn(el, W, sh);
    const sx = d3.scaleBand().domain(d3.range(periods.length)).range([0, W]).paddingInner(0.15);
    const sy = d3.scaleLinear().domain([0, d3.max(totals) || 1]).range([sh - 14, 4]);
    const winRect = strip.append("rect").attr("y", 0).attr("height", sh - 14).attr("fill", P.observed).attr("opacity", 0.12);
    const bars = strip.append("g").selectAll("rect").data(totals).join("rect").attr("x", (d, i) => sx(i)).attr("width", sx.bandwidth()).attr("y", d => sy(d)).attr("height", d => sh - 14 - sy(d)).attr("fill", P.null);
    let prevEra = null;
    periods.forEach((p, i) => {
      const e = L.era[p]; if (e == null || e === prevEra) return;
      if (prevEra !== null) strip.append("line").attr("x1", sx(i)).attr("x2", sx(i)).attr("y1", 0).attr("y2", sh - 14).attr("stroke", P.muted);
      strip.append("text").attr("x", sx(i) + 3).attr("y", sh - 3).attr("fill", P.muted).attr("font-size", 10).text(glabel(`${D.group_label} ${e}`));
      prevEra = e;
    });
    strip.append("g").selectAll("rect").data(periods).join("rect").attr("x", (d, i) => sx(i)).attr("width", sx.step()).attr("y", 0).attr("height", sh).attr("fill", "transparent")
      .on("pointermove", (e, d, i) => { const k = periods.indexOf(d); showTip(e, "week of " + d, [[T.mode === "help" ? "answers" : "mention weight", f2(totals[k])], ["agents active", (L.active[d] || []).length]]); })
      .on("pointerleave", hideTip).on("click", (e, d) => { T.i = periods.indexOf(d); slider.property("value", T.i); draw(); });

    // the circle
    const size = Math.min(W, 620), cx = W / 2, cy = size / 2, R = size / 2 - 64;
    const svg = svgIn(el, W, size);
    const angle = i => Math.PI / 2 - 2 * Math.PI * i / n, xy = i => [cx + R * Math.cos(angle(i)), cy - R * Math.sin(angle(i))];
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R).attr("fill", "none").attr("stroke", P.faint);
    const edgeLayer = svg.append("g"), nodeLayer = svg.append("g"), labelLayer = svg.append("g"), hitLayer = svg.append("g");

    function draw() {
      const lo = Math.max(0, T.i - T.win + 1), hi = T.i;
      stamp.text((T.win === 1 ? "week of " + periods[hi] : `${periods[lo]} to ${periods[hi]}`) + (L.era[periods[hi]] != null ? ` · ${glabel(`${D.group_label} ${L.era[periods[hi]]}`)}` : ""));
      winRect.attr("x", sx(lo)).attr("width", sx(hi) + sx.bandwidth() - sx(lo));
      bars.attr("fill", (d, i) => i === hi ? P.observed : P.null);
      const acc = new Map(), active = new Set(), msgs = new Map();
      for (let k = lo; k <= hi; k++) {
        (L[T.mode][periods[k]] || []).forEach(([s, t, w]) => acc.set(s * n + t, (acc.get(s * n + t) || 0) + w));
        (L.active[periods[k]] || []).forEach(a => active.add(a));
        Object.entries(L.messages[periods[k]] || {}).forEach(([a, m]) => msgs.set(+a, (msgs.get(+a) || 0) + m));
      }
      const edges = Array.from(acc, ([key, w]) => ({ s: Math.floor(key / n), t: key % n, w })).filter(e => e.s !== e.t).sort((a, b) => b.w - a.w);
      const shown = edges.slice(0, 80), wmax = shown.length ? shown[0].w : 1;
      const ins = new Map(), outs = new Map();
      edges.forEach(e => { ins.set(e.t, (ins.get(e.t) || 0) + e.w); outs.set(e.s, (outs.get(e.s) || 0) + e.w); });
      const inMax = d3.max(Array.from(ins.values())) || 1;
      const nodes = L.agents.map((a, i) => ({ i, id: a.id, family: a.family, on: active.has(i), in: ins.get(i) || 0, out: outs.get(i) || 0, msgs: msgs.get(i) || 0 }));
      const r = d => d.on ? 3.5 + 12 * Math.sqrt(d.in / inMax) : 2.5;
      const path = e => { const a = xy(e.s), b = xy(e.t), m = [(a[0] + b[0]) / 2 * 0.35 + cx * 0.65, (a[1] + b[1]) / 2 * 0.35 + cy * 0.65]; return `M${a[0]},${a[1]} Q${m[0]},${m[1]} ${b[0]},${b[1]}`; };
      edgeLayer.selectAll("path").data(shown, e => e.s + "-" + e.t).join("path").attr("d", path).attr("fill", "none").attr("stroke", P.ink).attr("stroke-linecap", "round")
        .attr("stroke-opacity", e => 0.08 + 0.45 * e.w / wmax).attr("stroke-width", e => 0.4 + 2.6 * e.w / wmax);
      nodeLayer.selectAll("circle").data(nodes, d => d.i).join("circle").attr("cx", d => xy(d.i)[0]).attr("cy", d => xy(d.i)[1]).attr("r", r)
        .attr("fill", d => d.on ? familyColour(d.family) : "#fff").attr("stroke", d => d.on ? "#fff" : P.axis).attr("stroke-width", d => d.on ? 2 : 1);
      const onCount = nodes.filter(d => d.on).length;
      const labelled = new Set(nodes.filter(d => d.on).sort((a, b) => b.in - a.in).slice(0, onCount <= 16 ? 16 : 8).map(d => d.i));
      labelLayer.selectAll("text").data(nodes.filter(d => labelled.has(d.i)), d => d.i).join("text").each(function (d) {
        const deg = (angle(d.i) * 180 / Math.PI) % 360, flip = deg > 90 && deg < 270 || deg < -90;
        d3.select(this).attr("transform", `translate(${xy(d.i)[0]},${xy(d.i)[1]}) rotate(${flip ? -deg + 180 : -deg})`)
          .attr("x", flip ? -(r(d) + 5) : r(d) + 5).attr("text-anchor", flip ? "end" : "start").attr("dominant-baseline", "middle").attr("font-size", 10).attr("fill", P.ink).text(d.id);
      });
      hitLayer.selectAll("circle").data(nodes, d => d.i).join("circle").attr("cx", d => xy(d.i)[0]).attr("cy", d => xy(d.i)[1]).attr("r", d => Math.max(12, r(d) + 4)).attr("fill", "transparent")
        .on("pointerenter", (e, d) => {
          edgeLayer.selectAll("path").attr("stroke", x => x.s === d.i || x.t === d.i ? P.observed : P.ink).attr("stroke-opacity", x => x.s === d.i || x.t === d.i ? 0.9 : 0.06);
          labelLayer.selectAll("text").attr("opacity", x => x.i === d.i || shown.some(y => (y.s === d.i && y.t === x.i) || (y.t === d.i && y.s === x.i)) ? 1 : 0.3);
        })
        .on("pointermove", (e, d) => showTip(e, d.id + " · " + d.family + (d.on ? "" : " · not active"), [[T.mode === "help" ? "answers received" : "mentions received", f2(d.in)], [T.mode === "help" ? "answers given" : "mentions sent", f2(d.out)], ["messages in window", int(d.msgs)]]))
        .on("pointerleave", () => { edgeLayer.selectAll("path").attr("stroke", P.ink).attr("stroke-opacity", e => 0.08 + 0.45 * e.w / wmax); labelLayer.selectAll("text").attr("opacity", 1); hideTip(); });
    }
    draw();
    familyLegend(el);
    note(el, "Agents sit in order of joining, clockwise from the top; hollow dots are agents not active in the window. Node size: ties received in the window. Click the strip or drag the slider to move in time.");
  }

  // ---- the reciprocity ladder: completeness per rung with its bootstrap interval
  function ladder(el) {
    const H = D.helping; if (!H) return;
    const rows = H.ladder.filter(r => r.model !== "no covariates");
    const W = width(el), left = 190, rh = 30, h = rh * rows.length + 36;
    const svg = svgIn(el, W, h);
    const x = d3.scaleLinear().domain([0, Math.max(1.05, d3.max(rows, r => r.ci_hi) || 1)]).range([left, W - 16]);
    const y = i => 10 + i * rh + rh / 2;
    grid(svg.append("g"), x, 5, h - 26, false);
    svg.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 4).attr("y2", h - 26).attr("stroke", P.axis);
    rows.forEach((r, i) => {
      const g = svg.append("g");
      g.append("text").attr("x", left - 10).attr("y", y(i)).attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 12).attr("fill", P.ink).text(r.model);
      g.append("line").attr("x1", x(r.ci_lo)).attr("x2", x(r.ci_hi)).attr("y1", y(i)).attr("y2", y(i)).attr("stroke", P.null).attr("stroke-width", 6).attr("stroke-linecap", "round");
      g.append("circle").attr("cx", x(r.completeness)).attr("cy", y(i)).attr("r", 5).attr("fill", r.model === "lookup table" ? P.muted : P.observed).attr("stroke", "#fff").attr("stroke-width", 2);
      g.append("rect").attr("x", 0).attr("y", y(i) - rh / 2).attr("width", W).attr("height", rh).attr("fill", "transparent")
        .on("pointermove", e => showTip(e, r.model, [["completeness", f2(r.completeness), P.observed], ["95% CI", `[${f2(r.ci_lo)}, ${f2(r.ci_hi)}]`, P.null], ["log loss", d3.format(".4f")(r.log_loss)]])).on("pointerleave", hideTip);
    });
    svg.append("g").attr("transform", `translate(0,${h - 26})`).call(d3.axisBottom(x).ticks(5).tickSize(3)).call(axis);
    svg.append("text").attr("x", W - 16).attr("y", h - 4).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("completeness: share of the lookup table's gain over the base rate");
  }

  // ---- the bystander curve: one panel per group, two series
  function bystander(el) {
    const H = D.helping; if (!H || !H.bystander.length) return;
    const sizes = H.sizes;
    const eras = Array.from(new Set(H.bystander.map(r => r.era))).filter(e => groupsShown().includes(`${D.group_label} ${e}`));
    if (!eras.length) return;
    const W = width(el), cols = Math.min(eras.length, W >= 700 ? 3 : W >= 480 ? 2 : 1), pw = W / cols, ph = 230;
    const svg = svgIn(el, W, ph * Math.ceil(eras.length / cols));
    const series = [["p_any_agent_responds", "anyone answers", P.series[0]], ["p_each_agent_responds", "a given agent answers", P.series[1]]];
    eras.forEach((era, k) => {
      const g = svg.append("g").attr("transform", `translate(${(k % cols) * pw},${Math.floor(k / cols) * ph})`);
      const rows = sizes.map(s => H.bystander.find(r => r.era === era && r.size === s)).filter(Boolean);
      const x = d3.scalePoint().domain(sizes).range([44, pw - 20]).padding(0.5), y = d3.scaleLinear().domain([0, 1]).range([ph - 40, 30]);
      g.append("text").attr("x", 8).attr("y", 16).attr("font-size", 12).attr("font-weight", 600).attr("fill", P.ink).text(glabel(`${D.group_label} ${era}`) + (thin.has(`${D.group_label} ${era}`) ? "  (thin)" : ""));
      grid(g.append("g").attr("transform", "translate(44,0)"), y, 4, pw - 64, true);
      g.append("g").attr("transform", "translate(44,0)").call(d3.axisLeft(y).ticks(4).tickSize(3).tickFormat(pct)).call(axis);
      g.append("g").attr("transform", `translate(0,${ph - 40})`).call(d3.axisBottom(x).tickSize(3)).call(axis);
      g.append("text").attr("x", (44 + pw - 20) / 2).attr("y", ph - 8).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", P.muted).text("agents active in the room that day");
      series.forEach(([col, label, colour]) => {
        const pts = rows.filter(r => r[col] != null);
        g.append("path").datum(pts).attr("fill", "none").attr("stroke", colour).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("d", d3.line().x(r => x(r.size)).y(r => y(r[col])));
        g.selectAll(null).data(pts).join("circle").attr("cx", r => x(r.size)).attr("cy", r => y(r[col])).attr("r", 4.5).attr("fill", colour).attr("stroke", "#fff").attr("stroke-width", 2);
        if (pts.length) { const last = pts[pts.length - 1], v = last[col]; g.append("text").attr("x", x(last.size)).attr("y", v > 0.85 ? y(v) + 17 : y(v) - 9).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", P.ink).text(pct(v)); }
      });
      g.selectAll(null).data(rows).join("rect").attr("x", r => x(r.size) - x.step() / 2).attr("y", 20).attr("width", x.step()).attr("height", ph - 60).attr("fill", "transparent")
        .on("pointermove", (e, r) => showTip(e, `${glabel(`${D.group_label} ${era}`)} · ${r.size} agents present`, series.map(([c, l, colour]) => [l, r[c] == null ? "n/a" : pct(r[c]), colour]).concat([["broadcast requests", int(r.requests)]]))).on("pointerleave", hideTip);
    });
    legend(el, series.map(([c, l, colour]) => [l, colour]), true);
  }

  // ---- hierarchy: steepness against its null band per group, then the ranking
  function hierarchy(el) {
    const Hh = D.hierarchy; if (!Hh) return;
    const rows = Hh.table.filter(r => r.statistic === "steepness" && r.null_lo != null && groupsShown().includes(r.group));
    if (!rows.length) return;
    const W = width(el), left = 110, rh = 30, h = rh * rows.length + 36;
    const svg = svgIn(el, W, h);
    const lo = d3.min(rows, r => Math.min(r.null_lo, r.observed)), hi = d3.max(rows, r => Math.max(r.null_hi, r.observed));
    const span = Math.max(hi - lo, 0.05);
    const x = d3.scaleLinear().domain([Math.max(0, lo - span * 0.2), hi + span * 0.25]).range([left, W - 16]);
    const y = i => 10 + i * rh + rh / 2;
    grid(svg.append("g"), x, 5, h - 26, false);
    rows.forEach((r, i) => {
      const g = svg.append("g");
      g.append("text").attr("x", left - 10).attr("y", y(i)).attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 12).attr("fill", thin.has(r.group) ? P.muted : P.ink).text(glabel(r.group));
      g.append("line").attr("x1", x(r.null_lo)).attr("x2", x(r.null_hi)).attr("y1", y(i)).attr("y2", y(i)).attr("stroke", P.null).attr("stroke-width", 7).attr("stroke-linecap", "round");
      g.append("circle").attr("cx", x(r.null_mean)).attr("cy", y(i)).attr("r", 3).attr("fill", P.muted);
      g.append("circle").attr("cx", x(r.observed)).attr("cy", y(i)).attr("r", 5.5).attr("fill", P.observed).attr("stroke", "#fff").attr("stroke-width", 2);
      g.append("text").attr("x", x(Math.max(r.observed, r.null_hi)) + 10).attr("y", y(i)).attr("dominant-baseline", "middle").attr("font-size", 10.5).attr("fill", P.muted).text(pfmt(r.p));
      g.append("rect").attr("x", 0).attr("y", y(i) - rh / 2).attr("width", W).attr("height", rh).attr("fill", "transparent")
        .on("pointermove", e => showTip(e, glabel(r.group), [["steepness", f3(r.observed), P.observed], ["null mean", f3(r.null_mean), P.null], ["null 95% band", `[${f3(r.null_lo)}, ${f3(r.null_hi)}]`], ["", pfmt(r.p)], ["directives", int(r.contests)], ["agents", int(r.agents)]])).on("pointerleave", hideTip);
    });
    svg.append("g").attr("transform", `translate(0,${h - 26})`).call(d3.axisBottom(x).ticks(5).tickSize(3)).call(axis);
    svg.append("text").attr("x", W - 16).attr("y", h - 4).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("steepness of the dominance hierarchy (0 flat, 1 linear)");
    legend(el, [["observed", P.observed], ["null 95% band and mean", P.null]]);
  }

  function ranks(el) {
    const Hh = D.hierarchy; if (!Hh || !Hh.ranks.length) return;
    const groups = groupsShown().filter(g => Hh.ranks.some(r => r.group === g));
    if (!groups.length) return;
    const W = width(el), cols = Math.min(groups.length, W >= 700 ? 3 : W >= 480 ? 2 : 1), pw = W / cols, top = 8, bh = 16, gap = 8;
    const per = groups.map(g => Hh.ranks.filter(r => r.group === g).slice(0, 10));
    const ph = top + 26 + d3.max(per, p => p.length) * (bh + gap) + 24;
    const svg = svgIn(el, W, ph * Math.ceil(groups.length / cols));
    const xmax = d3.max(per.flat(), r => r.davids_score) || 1;
    groups.forEach((gname, k) => {
      const rows = per[k], left = 112;
      const g = svg.append("g").attr("transform", `translate(${(k % cols) * pw},${Math.floor(k / cols) * ph})`);
      const x = d3.scaleLinear().domain([0, xmax * 1.15]).range([left, pw - 16]);
      g.append("text").attr("x", 8).attr("y", top + 10).attr("font-size", 12).attr("font-weight", 600).attr("fill", thin.has(gname) ? P.muted : P.ink).text(glabel(gname) + (thin.has(gname) ? "  (thin)" : ""));
      rows.forEach((r, i) => {
        const y0 = top + 26 + i * (bh + gap);
        g.append("text").attr("x", left - 8).attr("y", y0 + bh / 2).attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 11).attr("fill", P.ink).text(r.agent.length > 16 ? r.agent.slice(0, 15) + "…" : r.agent);
        endBar(g, x(0), x(r.davids_score), y0, bh).attr("fill", familyColour(D.families[r.agent]));
        g.append("text").attr("x", x(r.davids_score) + 5).attr("y", y0 + bh / 2).attr("dominant-baseline", "middle").attr("font-size", 10.5).attr("fill", P.ink).text(f2(r.davids_score));
        g.append("rect").attr("x", 0).attr("y", y0 - gap / 2).attr("width", pw).attr("height", bh + gap).attr("fill", "transparent")
          .on("pointermove", e => showTip(e, r.agent, [["David's score", f2(r.davids_score), familyColour(D.families[r.agent])], ["directives sent", int(r.directives_sent)], ["of which obeyed", int(Math.round(r.obeyed))], ["directives received", int(r.directives_received)], ["messages", int(r.messages)]])).on("pointerleave", hideTip);
      });
      g.append("g").attr("transform", `translate(0,${top + 26 + rows.length * (bh + gap)})`).call(d3.axisBottom(x).ticks(4).tickSize(3)).call(axis);
    });
    familyLegend(el);
    note(el, "Normalised David's score, top ten per group.");
  }

  // ---- the weekly series, stacked on one time axis, with a crosshair
  function trends(el) {
    const Tr = D.trends; if (!Tr || !Tr.weekly.length) return;
    const weeks = Tr.weekly.map(r => new Date(r.week)), W = width(el), left = 46, ph = 84, h = ph * Tr.series.length + 24;
    const svg = svgIn(el, W, h);
    const x = d3.scaleTime().domain(d3.extent(weeks)).range([left, W - 14]);
    const panels = Tr.series.map(([col, label], i) => {
      const vals = Tr.weekly.map(r => r[col]), rate = d3.max(vals.filter(v => v != null)) <= 1;
      const y = d3.scaleLinear().domain([0, rate ? 1 : (d3.max(vals) || 1)]).range([i * ph + ph - 14, i * ph + 26]);
      const g = svg.append("g");
      grid(g.append("g").attr("transform", `translate(${left},0)`), y, 2, W - left - 14, true);
      g.append("text").attr("x", left + 4).attr("y", i * ph + 12).attr("font-size", 11).attr("font-weight", 600).attr("fill", P.ink).text(label);
      g.append("g").attr("transform", `translate(${left},0)`).call(d3.axisLeft(y).ticks(2).tickSize(3).tickFormat(rate ? pct : d3.format("~s"))).call(axis);
      g.append("path").datum(Tr.weekly.filter(r => r[col] != null)).attr("fill", "none").attr("stroke", P.series[0]).attr("stroke-width", 2).attr("stroke-linejoin", "round")
        .attr("d", d3.line().defined(r => r[col] != null).x(r => x(new Date(r.week))).y(r => y(r[col])));
      return { col, label, y, rate };
    });
    Object.entries(Tr.era_starts).forEach(([era, start]) => {
      const xs = x(new Date(start));
      if (xs <= left + 1) return;  // the first group starts at the axis; a label there would sit on the panel title
      svg.append("line").attr("x1", xs).attr("x2", xs).attr("y1", 2).attr("y2", h - 24).attr("stroke", P.observed).attr("stroke-opacity", 0.6);
      svg.append("text").attr("x", xs + 4).attr("y", h - 28).attr("font-size", 10).attr("fill", P.observed).text(glabel(`${D.group_label} ${era}`));
    });
    svg.append("g").attr("transform", `translate(0,${h - 24})`).call(d3.axisBottom(x).ticks(Math.min(8, Math.floor(W / 110))).tickSize(3)).call(axis);
    const cross = svg.append("line").attr("y1", 2).attr("y2", h - 24).attr("stroke", P.muted).attr("stroke-width", 1).style("display", "none");
    const dots = svg.append("g").selectAll("circle").data(panels).join("circle").attr("r", 4).attr("fill", P.series[0]).attr("stroke", "#fff").attr("stroke-width", 2).style("display", "none");
    const bisect = d3.bisector(d => d).center;
    svg.append("rect").attr("x", left).attr("y", 0).attr("width", W - left - 14).attr("height", h - 24).attr("fill", "transparent")
      .on("pointermove", e => {
        const [mx] = d3.pointer(e); const k = bisect(weeks, x.invert(mx)); const r = Tr.weekly[k]; const xs = x(weeks[k]);
        cross.style("display", null).attr("x1", xs).attr("x2", xs);
        dots.style("display", p => r[p.col] == null ? "none" : null).attr("cx", xs).attr("cy", p => r[p.col] == null ? 0 : p.y(r[p.col]));
        showTip(e, "week of " + r.week + (r.era != null ? ` · ${glabel(`${D.group_label} ${r.era}`)}` : ""), panels.map(p => [p.label, r[p.col] == null ? "n/a" : p.rate ? pct(r[p.col]) : d3.format(".2~f")(r[p.col])]).concat([["messages", int(r.messages)]]));
      })
      .on("pointerleave", () => { cross.style("display", "none"); dots.style("display", "none"); hideTip(); });
  }

  // ---- diffusion: z per item, and the inferred network of who adopts after whom
  function diffusion_items(el) {
    const Df = D.diffusion; if (!Df || !Df.items.length) return;
    const z = Df.items.filter(r => r.z != null), W = width(el), h = 240, left = 40;
    const lo = Math.min(-3, d3.min(z, r => r.z) - 0.5), hi = Math.max(3, d3.max(z, r => r.z) + 0.5);
    const x = d3.scaleLinear().domain([lo, hi]).range([left, W - 14]);
    const bins = d3.bin().domain([lo, hi]).thresholds(x.ticks(30)).value(r => r.z)(z);
    const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).range([h - 34, 16]);
    const svg = svgIn(el, W, h);
    grid(svg.append("g").attr("transform", `translate(${left},0)`), y, 4, W - left - 14, true);
    svg.append("g").attr("transform", `translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickSize(3)).call(axis);
    svg.append("g").attr("transform", `translate(0,${h - 34})`).call(d3.axisBottom(x).ticks(8).tickSize(3)).call(axis);
    svg.selectAll("rect.bin").data(bins).join("rect").attr("class", "bin").attr("x", b => x(b.x0) + 1).attr("width", b => Math.max(1, x(b.x1) - x(b.x0) - 2)).attr("y", b => y(b.length)).attr("height", b => h - 34 - y(b.length)).attr("fill", P.null)
      .on("pointermove", (e, b) => showTip(e, `z in [${f2(b.x0)}, ${f2(b.x1)})`, [["items", b.length]].concat(b.slice(0, 8).map(r => [r.item, `z ${d3.format("+.2f")(r.z)}`])).concat(b.length > 8 ? [["…", `${b.length - 8} more`]] : []))).on("pointerleave", hideTip);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 10).attr("y2", h - 34).attr("stroke", P.muted);
    svg.append("line").attr("x1", x(1.645)).attr("x2", x(1.645)).attr("y1", 10).attr("y2", h - 34).attr("stroke", P.observed).attr("stroke-width", 1.5);
    svg.append("text").attr("x", x(1.645) + 4).attr("y", 14).attr("font-size", 10.5).attr("fill", P.observed).text("p < 0.05");
    svg.append("text").attr("x", W - 14).attr("y", h - 6).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("z of the observed adoption order against random orders, per item");
    svg.append("text").attr("x", 4).attr("y", 12).attr("font-size", 10.5).attr("fill", P.muted).text("items");
  }

  // the adoption network: credit flows from each adopter to those who had the item before it
  function diffusion_network(el) {
    const Df = D.diffusion; if (!Df || !Df.edges || !Df.edges.length) return;
    const led = new Map(), followed = new Map();
    Df.edges.forEach(e => { led.set(e.source, (led.get(e.source) || 0) + e.weight); followed.set(e.adopter, (followed.get(e.adopter) || 0) + e.weight); });
    const edges = Df.edges.slice().sort((a, b) => b.weight - a.weight).slice(0, 70);
    const ids = new Set(edges.flatMap(e => [e.source, e.adopter]));
    const lead = new Map(Df.leaders.map(r => [r.agent, r]));
    const nodes = Array.from(ids, id => ({ id, family: D.families[id] || "unknown", led: led.get(id) || 0, followed: followed.get(id) || 0, lead: lead.get(id) }));
    const W = width(el), h = Math.min(560, Math.max(360, W * 0.6));
    const svg = svgIn(el, W, h);
    drawForce(svg.append("g"), nodes, edges.map(e => ({ source: e.source, target: e.adopter, w: e.weight })), W, h,
      (edges.length < Df.edges.length ? `the ${edges.length} strongest of ${int(Df.edges.length)} edges` : `${edges.length} edges`) + `, ${nodes.length} agents`, true,
      d => [["credit received: others adopted after it", f2(d.led)], ["credit given: adopted after others", f2(d.followed)]].concat(d.lead ? [["lead score (+1 always first)", d3.format("+.2f")(d.lead.lead_score)], ["z against volume-weighted order", d.lead.z == null ? "n/a" : d3.format("+.1f")(d.lead.z)], ["", pfmt(d.lead.p)]] : []),
      d => d.led, l => [["credit from adopter to earlier adopter", f2(l.w)]]);
    familyLegend(el);
    note(el, "An arrow A → B: B adopted items after A had them, and B's credit for those items went partly to A. Node size: credit received. Hover an agent for its lead score.");
  }

  // each agent's lead score against the volume-weighted null
  function leaders(el) {
    const Df = D.diffusion; if (!Df || !Df.leaders || !Df.leaders.length) return;
    let rows = Df.leaders.filter(r => r.lead_score != null).sort((a, b) => b.lead_score - a.lead_score);
    let cut = false;
    if (rows.length > 30) { rows = rows.slice(0, 15).concat(rows.slice(-15)); cut = true; }
    const W = width(el), left = 150, rh = 22, h = rh * rows.length + 40;
    const svg = svgIn(el, W, h);
    const x = d3.scaleLinear().domain([-1, 1]).range([left, W - 16]);
    const y = i => 12 + i * rh + rh / 2;
    grid(svg.append("g"), x, 4, h - 28, false);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 4).attr("y2", h - 28).attr("stroke", P.muted);
    rows.forEach((r, i) => {
      const g = svg.append("g");
      const sd = r.z != null && r.z !== 0 ? Math.abs((r.lead_score - r.null_mean) / r.z) : null;
      g.append("text").attr("x", left - 10).attr("y", y(i)).attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 11).attr("fill", P.ink).text(r.agent.length > 20 ? r.agent.slice(0, 19) + "…" : r.agent);
      if (sd != null) g.append("line").attr("x1", x(Math.max(-1, r.null_mean - 1.96 * sd))).attr("x2", x(Math.min(1, r.null_mean + 1.96 * sd))).attr("y1", y(i)).attr("y2", y(i)).attr("stroke", P.null).attr("stroke-width", 6).attr("stroke-linecap", "round");
      if (r.null_mean != null) g.append("circle").attr("cx", x(r.null_mean)).attr("cy", y(i)).attr("r", 2.5).attr("fill", P.muted);
      g.append("circle").attr("cx", x(r.lead_score)).attr("cy", y(i)).attr("r", 5).attr("fill", familyColour(r.family || D.families[r.agent])).attr("stroke", "#fff").attr("stroke-width", 2);
      g.append("rect").attr("x", 0).attr("y", y(i) - rh / 2).attr("width", W).attr("height", rh).attr("fill", "transparent")
        .on("pointermove", e => showTip(e, r.agent + (r.family ? " · " + r.family : ""), [["lead score", d3.format("+.2f")(r.lead_score), familyColour(r.family || D.families[r.agent])], ["null mean, volume-weighted order", r.null_mean == null ? "n/a" : d3.format("+.2f")(r.null_mean), P.null], ["z, volume-weighted", r.z == null ? "n/a" : d3.format("+.1f")(r.z)], ["z, random order", r.z_random_order == null ? "n/a" : d3.format("+.1f")(r.z_random_order)], ["", pfmt(r.p)], ["adoption events", int(Math.round(r.led + r.followed))], ["messages", int(r.messages)]])).on("pointerleave", hideTip);
    });
    svg.append("g").attr("transform", `translate(0,${h - 28})`).call(d3.axisBottom(x).ticks(4).tickSize(3).tickFormat(d3.format("+.1f"))).call(axis);
    svg.append("text").attr("x", W - 16).attr("y", h - 6).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("lead score: +1 always first, −1 always after others; grey: 95% band under volume-weighted random order");
    familyLegend(el);
    if (cut) note(el, `The fifteen most leading and fifteen most following of ${Df.leaders.length} agents; the drawer below lists them all.`);
  }

  // ---- answer rates among undirected asks, by each factor
  function help_rates(el) {
    const U = D.helping && D.helping.undirected; if (!U) return;
    const factors = Array.from(new Set(U.rates.map(r => r.factor)));
    const W = width(el), cols = W >= 700 ? 4 : 2, pw = W / cols, bh = 16, gap = 10, left = 76;
    const rows = factors.map(f => U.rates.filter(r => r.factor === f));
    const ph = 52 + d3.max(rows, r => r.length) * (bh + gap) + 20;
    const svg = svgIn(el, W, ph * Math.ceil(factors.length / cols));
    const xmax = d3.max(U.rates, r => r.rate) || 0.1;
    factors.forEach((f, k) => {
      const g = svg.append("g").attr("transform", `translate(${(k % cols) * pw},${Math.floor(k / cols) * ph})`);
      const x = d3.scaleLinear().domain([0, xmax * 1.3]).range([left, pw - 14]);
      g.append("text").attr("x", 8).attr("y", 14).attr("font-size", 12).attr("font-weight", 600).attr("fill", P.ink).text(f);
      wrap(rows[k][0].label, Math.floor(pw / 6.5)).forEach((line, i) => g.append("text").attr("x", 8).attr("y", 28 + i * 12).attr("font-size", 10.5).attr("fill", P.muted).text(line));
      const top = 52;
      g.append("line").attr("x1", x(U.rate)).attr("x2", x(U.rate)).attr("y1", top - 4).attr("y2", top + rows[k].length * (bh + gap) - gap + 4).attr("stroke", P.axis);
      rows[k].forEach((r, i) => {
        const y0 = top + i * (bh + gap);
        g.append("text").attr("x", left - 6).attr("y", y0 + bh / 2).attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 11).attr("fill", P.ink).text(r.level);
        endBar(g, x(0), x(r.rate), y0, bh).attr("fill", P.series[0]);
        g.append("text").attr("x", x(r.rate) + 5).attr("y", y0 + bh / 2).attr("dominant-baseline", "middle").attr("font-size", 10.5).attr("fill", P.ink).text(pct(r.rate));
        g.append("rect").attr("x", 0).attr("y", y0 - gap / 2).attr("width", pw).attr("height", bh + gap).attr("fill", "transparent")
          .on("pointermove", e => showTip(e, `${f}: ${r.level}`, [["a given agent answers", pct(r.rate), P.series[0]], ["ask-agent pairs", int(r.n)]])).on("pointerleave", hideTip);
      });
    });
    note(el, `The share of undirected asks a given agent present answers, by each factor; the grey line is the overall rate, ${pct(U.rate)}.`);
  }

  // ---- diffusion: items by how many agents picked them up, and how fast an item reaches its adopters
  function diffusion_patterns(el) {
    const A = D.diffusion && D.diffusion.adoptions; if (!A) return;
    const W = width(el), two = W >= 640, pw = two ? W / 2 : W, h = 250;
    const svg = svgIn(el, W, two ? h : 2 * h);
    const kinds = ["link", "file", "term"].filter(k => A.kinds[k]);
    const kc = Object.fromEntries(kinds.map((k, i) => [k, P.series[i]]));
    const g1 = svg.append("g");
    const counts = d3.rollup(A.items, v => v.length, d => d.adopters, d => d.kind);
    const ns = Array.from(counts.keys()).sort((a, b) => a - b);
    const x1 = d3.scaleBand().domain(ns).range([44, pw - 14]).paddingInner(0.15);
    const y1 = d3.scaleLinear().domain([0, d3.max(ns, n => d3.sum(kinds, k => counts.get(n).get(k) || 0)) || 1]).range([h - 36, 24]);
    grid(g1.append("g").attr("transform", "translate(44,0)"), y1, 4, pw - 58, true);
    ns.forEach(n => {
      let acc = 0;
      kinds.forEach(k => {
        const c = counts.get(n).get(k) || 0; if (!c) return;
        g1.append("rect").attr("x", x1(n)).attr("width", x1.bandwidth()).attr("y", y1(acc + c)).attr("height", Math.max(0, y1(acc) - y1(acc + c) - 1)).attr("fill", kc[k])
          .on("pointermove", e => showTip(e, `${n} adopters`, kinds.map(kk => [kk + "s", int(counts.get(n).get(kk) || 0), kc[kk]]))).on("pointerleave", hideTip);
        acc += c;
      });
    });
    g1.append("g").attr("transform", "translate(44,0)").call(d3.axisLeft(y1).ticks(4).tickSize(3)).call(axis);
    g1.append("g").attr("transform", `translate(0,${h - 36})`).call(d3.axisBottom(x1).tickSize(3).tickValues(ns.filter((n, i) => ns.length <= 10 || i % 2 === 0))).call(axis);
    g1.append("text").attr("x", 44).attr("y", 14).attr("font-size", 11).attr("font-weight", 600).attr("fill", P.ink).text("items, by how many agents picked them up");
    g1.append("text").attr("x", pw - 14).attr("y", h - 8).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("agents that adopted the item");
    const g2 = svg.append("g").attr("transform", two ? `translate(${pw},0)` : `translate(0,${h})`);
    const x2 = d3.scaleLinear().domain([0, 30]).range([44, pw - 14]), y2 = d3.scaleLinear().domain([0, 1]).range([h - 36, 24]);
    const c = A.curve.filter(p => p.day <= 30);
    grid(g2.append("g").attr("transform", "translate(44,0)"), y2, 4, pw - 58, true);
    g2.append("path").datum(c).attr("fill", P.series[0]).attr("fill-opacity", 0.12).attr("d", d3.area().x(p => x2(p.day)).y0(p => y2(p.lo)).y1(p => y2(p.hi)));
    g2.append("path").datum(c).attr("fill", "none").attr("stroke", P.series[0]).attr("stroke-width", 2).attr("d", d3.line().x(p => x2(p.day)).y(p => y2(p.median)));
    g2.append("g").attr("transform", "translate(44,0)").call(d3.axisLeft(y2).ticks(4).tickSize(3).tickFormat(pct)).call(axis);
    g2.append("g").attr("transform", `translate(0,${h - 36})`).call(d3.axisBottom(x2).ticks(6).tickSize(3)).call(axis);
    g2.append("text").attr("x", 44).attr("y", 14).attr("font-size", 11).attr("font-weight", 600).attr("fill", P.ink).text("share of an item's adopters reached");
    g2.append("text").attr("x", pw - 14).attr("y", h - 8).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("days since the first use (median item, quartile band)");
    const bis = d3.bisector(p => p.day).center;
    g2.append("rect").attr("x", 44).attr("y", 20).attr("width", pw - 58).attr("height", h - 56).attr("fill", "transparent")
      .on("pointermove", e => { const [mx] = d3.pointer(e); const p = c[bis(c, x2.invert(mx))]; showTip(e, `${p.day} days after the first use`, [["the median item has reached", pct(p.median), P.series[0]], ["quartiles", `${pct(p.lo)} to ${pct(p.hi)}`]]); }).on("pointerleave", hideTip);
    legend(el, kinds.map(k => [k + "s", kc[k]]));
  }

  // ---- watch one item spread over the circle of agents
  const R = { item: null, step: null, timer: null };
  function diffusion_run(el) {
    const A = D.diffusion && D.diffusion.adoptions, L = D.timeline; if (!A || !L || !A.items.length) return;
    const agents = L.agents, n = agents.length;
    if (!R.item || !A.seq[R.item]) { R.item = A.items[0].item; R.step = null; }
    const ctl = d3.select(el).append("div").attr("class", "ctl");
    ctl.append("span").attr("class", "ctl-label").text("item");
    const stop = () => { if (R.timer) clearInterval(R.timer); R.timer = null; play.text("play"); };
    const sel = ctl.append("select").on("change", function () { R.item = this.value; R.step = null; stop(); draw(); });
    sel.selectAll("option").data(A.items).join("option").attr("value", d => d.item).property("selected", d => d.item === R.item).text(d => `${d.item}  (${d.kind}, ${d.adopters} agents, ${d.days} days)`);
    const play = ctl.append("button").attr("class", "seg").text("play").on("click", () => {
      if (R.timer) { stop(); return; }
      R.step = 0; draw(); play.text("pause");
      R.timer = setInterval(() => { if (R.step >= A.seq[R.item].length) { stop(); return; } R.step += 1; draw(); }, 800);
    });
    const stamp = ctl.append("span").attr("class", "stamp");
    const slider = d3.select(el).append("input").attr("type", "range").attr("class", "slider").attr("min", 0).on("input", function () { R.step = +this.value; stop(); draw(); });
    const tie = new Map();  // mentions over the whole record, between pairs of agents
    Object.values(L.mentions).forEach(list => list.forEach(([s, t, w]) => { const k = Math.min(s, t) * n + Math.max(s, t); tie.set(k, (tie.get(k) || 0) + w); }));
    const W = width(el), size = Math.min(W, 560), cx = W / 2, cy = size / 2, Rr = size / 2 - 64;
    const svg = svgIn(el, W, size);
    const angle = i => Math.PI / 2 - 2 * Math.PI * i / n, xy = i => [cx + Rr * Math.cos(angle(i)), cy - Rr * Math.sin(angle(i))];
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", Rr).attr("fill", "none").attr("stroke", P.faint);
    const edgeLayer = svg.append("g"), nodeLayer = svg.append("g"), labelLayer = svg.append("g");
    const list = d3.select(el).append("ol").attr("class", "runlist");
    function draw() {
      const seq = A.seq[R.item];
      if (R.step === null) R.step = seq.length;
      const k = Math.min(R.step, seq.length), have = seq.slice(0, k);
      slider.attr("max", seq.length).property("value", k);
      stamp.text(k ? `day ${d3.format(".1f")(have[k - 1][1])} · ${k} of ${seq.length} agents` : `${seq.length} agents will pick it up`);
      const edges = [];
      have.forEach((a, i) => have.slice(i + 1).forEach(b => { const w = tie.get(Math.min(a[0], b[0]) * n + Math.max(a[0], b[0])); if (w) edges.push({ s: a[0], t: b[0], w }); }));
      const wmax = d3.max(edges, e => e.w) || 1;
      const path = e => { const a = xy(e.s), b = xy(e.t), m = [(a[0] + b[0]) / 2 * 0.35 + cx * 0.65, (a[1] + b[1]) / 2 * 0.35 + cy * 0.65]; return `M${a[0]},${a[1]} Q${m[0]},${m[1]} ${b[0]},${b[1]}`; };
      edgeLayer.selectAll("path").data(edges, e => e.s + "-" + e.t).join("path").attr("d", path).attr("fill", "none").attr("stroke", P.ink).attr("stroke-linecap", "round").attr("stroke-opacity", e => 0.1 + 0.5 * e.w / wmax).attr("stroke-width", e => 0.5 + 2.5 * e.w / wmax);
      const order = new Map(have.map((s, i) => [s[0], i]));
      nodeLayer.selectAll("circle").data(agents.map((a, i) => ({ i, id: a.id, family: a.family })), d => d.i).join("circle").attr("cx", d => xy(d.i)[0]).attr("cy", d => xy(d.i)[1])
        .attr("r", d => order.has(d.i) ? (order.get(d.i) === k - 1 ? 11 : 7) : 2.5).attr("fill", d => order.has(d.i) ? familyColour(d.family) : "#fff").attr("stroke", d => order.has(d.i) ? "#fff" : P.axis).attr("stroke-width", d => order.has(d.i) ? 2 : 1);
      labelLayer.selectAll("text").data(have.map(s => s[0]), d => d).join("text").each(function (i) {
        const deg = (angle(i) * 180 / Math.PI) % 360, flip = deg > 90 && deg < 270 || deg < -90;
        d3.select(this).attr("transform", `translate(${xy(i)[0]},${xy(i)[1]}) rotate(${flip ? -deg + 180 : -deg})`).attr("x", flip ? -15 : 15).attr("text-anchor", flip ? "end" : "start").attr("dominant-baseline", "middle").attr("font-size", 10).attr("fill", P.ink).text(`${order.get(i) + 1}. ${agents[i].id}`);
      });
      list.selectAll("li").data(seq).join("li").attr("class", (s, i) => i < k ? "on" : null).text((s, i) => `${agents[s[0]].id} · day ${d3.format(".1f")(s[1])}`);
    }
    draw();
    familyLegend(el);
  }

  // ---- lead score in adoption against David's score in the hierarchy
  function lead_vs_rank(el) {
    const LR = D.diffusion && D.diffusion.lead_rank; if (!LR) return;
    const W = width(el), h = 300, left = 50;
    const svg = svgIn(el, W, h);
    const ext = d3.extent(LR.points, p => p.rank), pad = (ext[1] - ext[0]) * 0.15 || 1;  // David's scores sit well above zero; show their spread
    const x = d3.scaleLinear().domain([-1, 1]).range([left, W - 16]), y = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([h - 40, 16]);
    grid(svg.append("g").attr("transform", `translate(${left},0)`), y, 4, W - left - 16, true);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 16).attr("y2", h - 40).attr("stroke", P.muted);
    svg.append("g").attr("transform", `translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickSize(3)).call(axis);
    svg.append("g").attr("transform", `translate(0,${h - 40})`).call(d3.axisBottom(x).ticks(4).tickSize(3).tickFormat(d3.format("+.1f"))).call(axis);
    svg.append("text").attr("x", W - 16).attr("y", h - 8).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", P.muted).text("lead score in adoption: +1 always first, −1 always after others");
    svg.append("text").attr("x", 4).attr("y", 12).attr("font-size", 10.5).attr("fill", P.muted).text("David's score");
    svg.append("g").selectAll("circle").data(LR.points).join("circle").attr("cx", p => x(p.lead)).attr("cy", p => y(p.rank)).attr("r", 5.5).attr("fill", p => familyColour(p.family)).attr("stroke", "#fff").attr("stroke-width", 2);
    const sorted = LR.points.slice().sort((a, b) => b.lead - a.lead), named = new Set(sorted.slice(0, 3).concat(sorted.slice(-3)).map(p => p.agent));
    svg.append("g").selectAll("text").data(LR.points.filter(p => named.has(p.agent))).join("text").attr("x", p => Math.min(W - 70, x(p.lead) + 7)).attr("y", p => y(p.rank) - 6).attr("font-size", 10).attr("fill", P.ink).text(p => p.agent);
    svg.append("g").selectAll("circle").data(LR.points).join("circle").attr("cx", p => x(p.lead)).attr("cy", p => y(p.rank)).attr("r", 12).attr("fill", "transparent")
      .on("pointermove", (e, p) => showTip(e, p.agent + " · " + p.family, [["lead score", d3.format("+.2f")(p.lead), familyColour(p.family)], ["David's score", f2(p.rank)], ["messages", int(p.messages)]])).on("pointerleave", hideTip);
    familyLegend(el);
  }

  // ---- wiring
  const RENDER = { nulls, network, timeline, ladder, bystander, hierarchy, ranks, trends, diffusion_items, diffusion_network, leaders, help_rates, diffusion_patterns, diffusion_run, lead_vs_rank };
  function renderAll(groupedOnly) {
    document.querySelectorAll(".fig").forEach(el => {
      if (groupedOnly && !el.dataset.groups) return;
      const f = RENDER[el.dataset.fig]; if (!f) return;
      el.replaceChildren();
      try { f(el); } catch (err) { el.textContent = "figure failed: " + err.message; console.error(el.dataset.fig, err); }
    });
  }
  const bar = document.getElementById("groupbar");
  if (bar && D.groups.length > 1) {
    const b = d3.select(bar);
    b.append("span").attr("class", "ctl-label").text("show");
    const opts = [{ name: "all", label: "all " + D.group_label + "s" }].concat(D.groups.map(g => ({ name: g.name, label: glabel(g.name), thin: g.thin, title: g.thin ? `${g.median_active} agents a day on average: too thin to judge` : `${g.median_active} agents a day on average` })));
    b.selectAll("button").data(opts).join("button").attr("class", o => "seg" + (o.thin ? " thin" : "") + (o.name === state.group ? " on" : "")).attr("title", o => o.title || null).text(o => o.label)
      .on("click", function (e, o) { state.group = o.name; b.selectAll("button").classed("on", x => x.name === o.name); renderAll(true); });
  }
  renderAll(false);
  let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => renderAll(false), 250); });
})();
