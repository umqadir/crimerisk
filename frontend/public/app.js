/* CrimeRisk public map.
 *
 * One bootstrap manifest, three PMTiles archives (county / tract / block
 * group) and GEOID-keyed JSON lookup shards. No hidden map instances, no
 * runtime CDN, no cache-busting timestamps.
 */
(function () {
  "use strict";

  var T0 = performance.now();
  var M = null;            // bootstrap manifest
  var map = null;
  var basemapMerged = false;

  var state = {
    measure: "exposure",
    crime: "ov",
    sel: null,             // {level, geoid}
    cmp: null,             // {level, geoid}
    tractView: null,       // block-group geoid whose tract estimate was asked for
    marker: null
  };

  var shardIndex = {};     // level -> {base_depth, split:Set}
  var shardIndexReady = {};// level -> Promise
  var shardCache = {};     // level -> prefix -> Promise
  var benchCache = {};     // state -> Promise
  var LEVELS = { c: "county", t: "tract", b: "bg" };
  var SRC = { county: "c", tract: "t", bg: "b" };

  var $ = function (id) { return document.getElementById(id); };

  /* ------------------------------------------------------------ helpers -- */

  function fmt(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return Math.round(v).toLocaleString("en-US");
  }

  function legendClass(v) {
    if (v === null || v === undefined || isNaN(v)) return null;
    for (var i = M.legend.length - 1; i >= 0; i--) {
      if (v >= M.legend[i].lo) return M.legend[i];
    }
    return M.legend[0];
  }

  function field() {
    return (state.measure === "exposure" ? "x_" : "r_") + state.crime;
  }

  function measureIndex() {
    return M.measure_order.indexOf(field());
  }

  /* What the selected number is, per index and measure (manifest crime_help); falls back to the
     measure's generic line. */
  function helpText(crime, measure) {
    var h = M.crime_help && M.crime_help[crime];
    if (h && h[measure]) return h[measure];
    return M.measures.filter(function (m) { return m.key === measure; })[0].subcopy;
  }

  function crimeDef(key) {
    for (var i = 0; i < M.crimes.length; i++) if (M.crimes[i].key === key) return M.crimes[i];
    return M.crimes[0];
  }

  function measureLabel() {
    var d = crimeDef(state.crime);
    if (d.composite) {
      return d.label + (state.measure === "exposure" ? " crime exposure" : " crime per resident");
    }
    return d.label + (state.measure === "exposure" ? " exposure" : " per resident");
  }

  function tractOnly() { return !!crimeDef(state.crime).tract_only; }

  /* ------------------------------------------------------------- shards -- */

  function shardPrefix(level, geoid) {
    var idx = shardIndex[level];
    var d = idx.base_depth;
    while (idx.split.has(geoid.slice(0, d)) && d < geoid.length) d++;
    return geoid.slice(0, d);
  }

  /* A failed request is a load error, never "no data": the promise rejects and
     is dropped from the cache so the next attempt refetches. */
  function loadShardIndex(level) {
    if (!shardIndexReady[level]) {
      shardIndexReady[level] = fetch("data/shards/" + level + "/index.json")
        .then(function (r) {
          if (!r.ok) throw new Error("shard index " + r.status);
          return r.json();
        })
        .then(function (j) {
          shardIndex[level] = { base_depth: j.base_depth, split: new Set(j.split) };
        })
        .catch(function (e) { delete shardIndexReady[level]; throw e; });
    }
    return shardIndexReady[level];
  }

  /* Resolves to the area's record, or null when the published tree has no record
     for it (a 404 shard is an absent prefix, i.e. no data). Rejects when the data
     could not be fetched. */
  function lookup(level, geoid) {
    return loadShardIndex(level).then(function () {
      var prefix = shardPrefix(level, geoid);
      var cache = shardCache[level] || (shardCache[level] = {});
      if (!cache[prefix]) {
        cache[prefix] = fetch("data/shards/" + level + "/" + prefix + ".json")
          .then(function (r) {
            if (r.status === 404) return {};
            if (!r.ok) throw new Error("shard " + r.status);
            return r.json();
          })
          .catch(function (e) { delete cache[prefix]; throw e; });
      }
      return cache[prefix].then(function (obj) { return obj[geoid] || null; });
    });
  }

  /* ------------------------------------------------------------- status -- */

  /* Publication status per measure (manifest status.codes): 0 published,
     1 suppressed below a denominator or support floor, 2 murder / rape at block
     group, 3 no data. Shard records carry the 20 codes at position 11; tiles carry
     them packed in one number, digit i (base status.base) for measure i. */
  var ST = { PUB: 0, SUP: 1, TRACT: 2, NONE: 3 };

  function recStatus(rec, mi) {
    if (!rec) return ST.NONE;
    var v = rec[6] ? rec[6][mi] : null;
    if (v !== null && v !== undefined) return ST.PUB;
    var a = rec[11];
    return a && a[mi] !== null && a[mi] !== undefined ? a[mi] : ST.NONE;
  }

  function statusDigit(mi) {
    return ["%", ["floor", ["/", ["coalesce", ["get", M.status.tile_field], 0],
      Math.pow(M.status.base, mi)]], M.status.base];
  }

  function tileStatus(props, mi) {
    var ms = props[M.status.tile_field] || 0;
    return Math.floor(ms / Math.pow(M.status.base, mi)) % M.status.base;
  }

  /* The card sentence for an unpublished value. */
  function statusText(r) {
    var c = M.status.card;
    if (r.error) return c.load_error;
    if (r.status === ST.SUP) {
      if (r.level === "county") return c.suppressed_county;
      if (state.measure === "resident") return c.suppressed_resident;
      return crimeDef(state.crime).composite ? c.suppressed_exposure_composite : c.suppressed_exposure;
    }
    if (r.status === ST.TRACT) return tractOnlyText();
    return c.no_data;
  }

  function statusShort(r) {
    if (r.error) return "Not loaded";
    if (r.status === ST.SUP) return "Not published";
    return "No estimate";
  }

  function tractOnlyText() {
    return M.status.card.tract_only.replace("{offense}", crimeDef(state.crime).label);
  }

  function bench(st) {
    if (!benchCache[st]) {
      benchCache[st] = fetch("data/bench/" + st + ".json")
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { return null; });
    }
    return benchCache[st];
  }

  /* ---------------------------------------------------------------- URL -- */

  function writeUrl() {
    var p = new URLSearchParams();
    p.set("m", state.measure);
    p.set("x", state.crime);
    if (state.sel) p.set("s", state.sel.level + ":" + state.sel.geoid);
    if (state.cmp) p.set("k", state.cmp.level + ":" + state.cmp.geoid);
    if (map) {
      var c = map.getCenter();
      p.set("v", map.getZoom().toFixed(2) + "/" + c.lat.toFixed(4) + "/" + c.lng.toFixed(4));
    }
    history.replaceState(null, "", "?" + p.toString());
  }

  function parseRef(v) {
    if (!v) return null;
    var bits = v.split(":");
    if (bits.length !== 2) return null;
    if (["county", "tract", "bg"].indexOf(bits[0]) < 0) return null;
    if (!/^[0-9]{5,12}$/.test(bits[1])) return null;
    return { level: bits[0], geoid: bits[1] };
  }

  function readUrl() {
    var p = new URLSearchParams(location.search);
    if (p.get("m") === "resident" || p.get("m") === "exposure") state.measure = p.get("m");
    if (p.get("x")) {
      for (var i = 0; i < M.crimes.length; i++) if (M.crimes[i].key === p.get("x")) state.crime = p.get("x");
    }
    state.sel = parseRef(p.get("s"));
    state.cmp = parseRef(p.get("k"));
    var v = p.get("v");
    if (v) {
      var bits = v.split("/").map(Number);
      if (bits.length === 3 && bits.every(function (n) { return !isNaN(n); })) {
        return { zoom: bits[0], center: [bits[2], bits[1]] };
      }
    }
    return null;
  }

  /* ------------------------------------------------------------- legend -- */

  /* An area without a value paints the neutral grey on every layer, so a coarser
     layer underneath never shows through as if it were this area's value. */
  function paintExpression() {
    var expr = ["step", ["get", field()], M.legend[0].color];
    for (var i = 1; i < M.legend.length; i++) expr.push(M.legend[i].lo, M.legend[i].color);
    return ["case", ["has", field()], expr, M.no_estimate.color];
  }

  /* Areas whose value for the active measure is suppressed carry the hatch. */
  function suppressedFilter() {
    return ["all", ["!", ["has", field()]], ["==", statusDigit(measureIndex()), ST.SUP]];
  }

  function legendItem(ul, swatchClass, color, text) {
    var li = document.createElement("li");
    var sw = document.createElement("span");
    sw.className = swatchClass;
    if (color) sw.style.backgroundColor = color;
    var lab = document.createElement("span");
    lab.textContent = text;
    li.appendChild(sw);
    li.appendChild(lab);
    ul.appendChild(li);
  }

  /* Compact legend for the collapsed mobile sheet: the seven classes, then the
     unpublished swatch. */
  function renderLegendStrip() {
    var el = $("legend-strip");
    if (!el) return;
    el.innerHTML = "";
    var bar = document.createElement("span");
    bar.className = "bar";
    M.legend.forEach(function (c) {
      var b = document.createElement("span");
      b.style.background = c.color;
      b.title = c.label;
      bar.appendChild(b);
    });
    var lo = document.createElement("span");
    lo.className = "end";
    lo.textContent = "<" + M.legend[1].lo;
    var hi = document.createElement("span");
    hi.className = "end";
    hi.textContent = M.legend[M.legend.length - 1].lo + "+";
    var sup = document.createElement("span");
    sup.className = "swatch sup";
    sup.style.backgroundColor = M.no_estimate.color;
    sup.title = M.status.legend.suppressed_low_denominator;
    var supl = document.createElement("span");
    supl.className = "end";
    supl.textContent = "Not published";
    el.appendChild(lo);
    el.appendChild(bar);
    el.appendChild(hi);
    el.appendChild(sup);
    el.appendChild(supl);
  }

  function renderLegend() {
    var ul = $("legend");
    ul.innerHTML = "";
    M.legend.forEach(function (c) {
      var li = document.createElement("li");
      var sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = c.color;
      var lab = document.createElement("span");
      lab.textContent = c.label;
      var rng = document.createElement("span");
      rng.className = "range";
      rng.textContent = c.hi === null ? c.lo + "+" : c.lo + "–" + (c.hi - 1);
      li.appendChild(sw);
      li.appendChild(lab);
      li.appendChild(rng);
      ul.appendChild(li);
    });
    // Unpublished areas, at every zoom.
    legendItem(ul, "swatch sup", M.no_estimate.color, M.status.legend.suppressed_low_denominator);
    legendItem(ul, "swatch", M.no_estimate.color, M.status.legend.no_data);
    if ($("special-toggle").checked) {
      var li2 = document.createElement("li");
      var sw2 = document.createElement("span");
      sw2.className = "swatch hatch";
      var l2 = document.createElement("span");
      l2.textContent = "Special-use area";
      li2.appendChild(sw2);
      li2.appendChild(l2);
      ul.appendChild(li2);
    }
  }

  /* -------------------------------------------------------------- chips -- */

  function renderChoices() {
    var cr = $("crimes");
    cr.innerHTML = "";
    M.crimes.forEach(function (c, i) {
      if (i === 3) {
        var sep = document.createElement("span");
        sep.className = "sep";
        cr.appendChild(sep);
      }
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = c.label;
      b.setAttribute("aria-pressed", String(c.key === state.crime));
      b.addEventListener("click", function () { setCrime(c.key); });
      cr.appendChild(b);
    });

    var me = $("measures");
    me.innerHTML = "";
    M.measures.forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = m.label;
      b.setAttribute("aria-pressed", String(m.key === state.measure));
      b.addEventListener("click", function () { setMeasure(m.key); });
      me.appendChild(b);
    });
    $("subcopy").textContent = helpText(state.crime, state.measure);
  }

  /* ---------------------------------------------------------------- map -- */

  function addChoropleth() {
    ["county", "tract", "bg"].forEach(function (level) {
      var t = M.tiles[level];
      var src = { type: "vector", url: "pmtiles://" + t.url, promoteId: "geoid" };
      if (level === "bg") src.attribution = M.attribution;
      map.addSource(SRC[level], src);
    });

    var paint = paintExpression();
    var supFilter = suppressedFilter();

    // Each fill is followed by its suppression hatch, so a finer layer covers the
    // coarser layer's hatch as well as its colour. Counties below the support
    // floor, and cells below a denominator floor, are grey and hatched: not a
    // hole and not a colour class.
    map.addLayer({
      id: "fill-county", type: "fill", source: "c", "source-layer": M.tiles.county.layer,
      maxzoom: M.zoom.bg_min,
      paint: { "fill-color": paint, "fill-opacity": 1 }
    });
    map.addLayer({
      id: "sup-county", type: "fill", source: "c", "source-layer": M.tiles.county.layer,
      maxzoom: M.zoom.bg_min, filter: supFilter,
      paint: { "fill-pattern": "hatch-sup" }
    });
    map.addLayer({
      id: "fill-tract", type: "fill", source: "t", "source-layer": M.tiles.tract.layer,
      minzoom: M.zoom.tract_min,
      paint: { "fill-color": paint, "fill-opacity": 1 }
    });
    map.addLayer({
      id: "sup-tract", type: "fill", source: "t", "source-layer": M.tiles.tract.layer,
      minzoom: M.zoom.tract_min, filter: supFilter,
      paint: { "fill-pattern": "hatch-sup" }
    });
    map.addLayer({
      id: "fill-bg", type: "fill", source: "b", "source-layer": M.tiles.bg.layer,
      minzoom: M.zoom.bg_min,
      paint: { "fill-color": paint, "fill-opacity": 1 }
    });
    map.addLayer({
      id: "sup-bg", type: "fill", source: "b", "source-layer": M.tiles.bg.layer,
      minzoom: M.zoom.bg_min, filter: supFilter,
      paint: { "fill-pattern": "hatch-sup" }
    });

    map.addLayer({
      id: "hatch-tract", type: "fill", source: "t", "source-layer": M.tiles.tract.layer,
      minzoom: M.zoom.tract_min, filter: [">", ["coalesce", ["get", "su"], 0], 0],
      layout: { visibility: "none" },
      paint: { "fill-pattern": "hatch", "fill-opacity": 1 }
    });
    map.addLayer({
      id: "hatch-bg", type: "fill", source: "b", "source-layer": M.tiles.bg.layer,
      minzoom: M.zoom.bg_min, filter: [">", ["coalesce", ["get", "su"], 0], 0],
      layout: { visibility: "none" },
      paint: { "fill-pattern": "hatch", "fill-opacity": 1 }
    });

    // Charcoal boundaries at neighborhood zoom.
    map.addLayer({
      id: "edge-tract", type: "line", source: "t", "source-layer": M.tiles.tract.layer,
      minzoom: 7,
      paint: {
        "line-opacity": 1,
        "line-color": "rgba(34,36,40,0.38)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.3, 12, 0.8]
      }
    });
    map.addLayer({
      id: "edge-bg", type: "line", source: "b", "source-layer": M.tiles.bg.layer,
      minzoom: M.zoom.bg_min,
      paint: {
        "line-color": "rgba(34,36,40,0.30)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.25, 12, 0.7]
      }
    });

    // Hover and selection outlines.
    ["county", "tract", "bg"].forEach(function (level) {
      map.addLayer({
        id: "hover-" + level, type: "line", source: SRC[level],
        "source-layer": M.tiles[level].layer,
        filter: ["==", ["get", "geoid"], ""],
        paint: { "line-color": "#16181c", "line-width": 1.4 }
      });
      map.addLayer({
        id: "selhalo-" + level, type: "line", source: SRC[level],
        "source-layer": M.tiles[level].layer,
        filter: ["==", ["get", "geoid"], ""],
        paint: { "line-color": "#ffffff", "line-width": 5 }
      });
      map.addLayer({
        id: "sel-" + level, type: "line", source: SRC[level],
        "source-layer": M.tiles[level].layer,
        filter: ["==", ["get", "geoid"], ""],
        paint: { "line-color": "#16181c", "line-width": 2.2 }
      });
    });
  }

  /* Merge the OpenFreeMap positron basemap once it arrives: context below the
     choropleth, water, roads, borders and every label above it. The map works
     without it, so a basemap outage never produces a blank page. */
  /* Basemap layers that belong ON TOP of the choropleth: water (the mask that
     keeps lakes and harbors out of the data), the major road skeleton, and
     administrative borders. Minor streets, landuse and buildings stay below so
     the color reads as one surface. */
  var ABOVE = /^(water|waterway|boundary_|highway_motorway|railway$)/;
  /* The motorway casing and the wide "subtle" underlay are white and several
     pixels across; over the choropleth they read as gashes. Only the inner
     line is hoisted, restyled to one thin light-grey stroke. */
  var SKIP_ABOVE = /^highway_motorway(_bridge)?_(casing|subtle)$/;
  var MOTORWAY_LINE = {
    "line-color": "rgba(244,242,238,0.9)",
    "line-blur": 0,
    "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 9, 0.9, 12, 1.6, 16, 2.6]
  };

  function mergeBasemap(style) {
    if (basemapMerged) return;
    if (!map.isStyleLoaded()) { setTimeout(function () { mergeBasemap(style); }, 120); return; }
    basemapMerged = true;
    try { map.setSprite(style.sprite || M.basemap.sprite); } catch (e) { /* icons are optional */ }
    Object.keys(style.sources).forEach(function (id) {
      if (!map.getSource(id)) map.addSource(id, style.sources[id]);
    });
    // Bottom of our stack is the county fill; basemap context goes under it.
    var firstFill = "fill-county";
    var firstOutline = "hover-county";
    style.layers.forEach(function (l) {
      if (map.getLayer(l.id)) return;
      try {
        if (l.type === "background") { map.addLayer(l, firstFill); return; }
        if (l.type === "symbol") { map.addLayer(l); return; }
        if (SKIP_ABOVE.test(l.id)) { map.addLayer(l, firstFill); return; }
        if (/^highway_motorway/.test(l.id)) {
          l = JSON.parse(JSON.stringify(l));
          l.paint = Object.assign({}, l.paint, MOTORWAY_LINE);
          delete l.paint["line-gap-width"];
          delete l.paint["line-opacity"];
        }
        if (l.id === "water") {
          // Water sits on top of the choropleth as the mask. It is a light,
          // near-neutral cool grey: lighter than both pale blue classes and
          // cooler than the warm near-neutral middle, so it sits outside the
          // ramp at both ends without making the ocean heavier than the
          // country. Water also carries no cell boundary, which is the other
          // cue that separates a lake from a low-crime block group. CIELab
          // distance is 21.1 to the darkest blue class, 7.8 to the palest, 7.8
          // to the near-neutral middle and 6.5 to the basemap land.
          l = JSON.parse(JSON.stringify(l));
          l.paint = l.paint || {};
          l.paint["fill-color"] = "#e2eaee";
        }
        if (ABOVE.test(l.id)) map.addLayer(l, firstOutline);
        else map.addLayer(l, firstFill);
      } catch (e) { /* a basemap layer we cannot place is not fatal */ }
    });
  }

  /* Transparent diagonal hatch. The pattern must not hide the colour underneath,
     so it is thin dark strokes on a fully transparent tile. The special-use
     annotation leans one way; the suppression hatch leans the other and is
     lighter, drawn over the grey of an unpublished area. */
  function hatchImage(mirror, stroke) {
    var s = 16, c = document.createElement("canvas");
    c.width = c.height = s;
    var g = c.getContext("2d");
    g.clearRect(0, 0, s, s);
    if (mirror) { g.translate(s, 0); g.scale(-1, 1); }
    g.strokeStyle = stroke || "rgba(20,22,26,0.62)";
    g.lineWidth = 2;
    g.lineCap = "square";
    g.beginPath();
    g.moveTo(-s, s); g.lineTo(s, -s);
    g.moveTo(0, 2 * s); g.lineTo(2 * s, 0);
    g.moveTo(-s, 2 * s); g.lineTo(2 * s, -s);
    g.stroke();
    var px = g.getImageData(0, 0, s, s).data;
    return { width: s, height: s, data: new Uint8Array(px.buffer.slice(0)) };
  }

  /* ------------------------------------------------------------- update -- */

  function activeFillLayer() {
    var z = map.getZoom();
    if (z < M.zoom.tract_min) return "fill-county";
    if (tractOnly() || z < M.zoom.bg_min) return "fill-tract";
    return "fill-bg";
  }

  function activeLevel() {
    var l = activeFillLayer();
    return l === "fill-county" ? "county" : (l === "fill-tract" ? "tract" : "bg");
  }

  function repaint() {
    var paint = paintExpression();
    ["fill-county", "fill-tract", "fill-bg"].forEach(function (id) {
      map.setPaintProperty(id, "fill-color", paint);
    });
    var supFilter = suppressedFilter();
    ["sup-county", "sup-tract", "sup-bg"].forEach(function (id) {
      map.setFilter(id, supFilter);
    });
    var rare = tractOnly();
    ["fill-bg", "sup-bg", "edge-bg"].forEach(function (id) {
      map.setLayoutProperty(id, "visibility", rare ? "none" : "visible");
    });
    // Tract boundaries belong to whichever zooms the tract layer owns.
    map.setPaintProperty("edge-tract", "line-opacity",
      rare ? 1 : ["step", ["zoom"], 1, M.zoom.bg_min, 0]);
    updateHatch();
    updateNote();
    renderSheetSummary();
  }

  function updateHatch() {
    var on = $("special-toggle").checked;
    var rare = tractOnly();
    if (map.getLayer("hatch-tract")) {
      map.setLayoutProperty("hatch-tract", "visibility", on ? "visible" : "none");
      map.setPaintProperty("hatch-tract", "fill-opacity",
        rare ? 1 : ["step", ["zoom"], 1, M.zoom.bg_min, 0]);
    }
    if (map.getLayer("hatch-bg")) {
      map.setLayoutProperty("hatch-bg", "visibility", on && !rare ? "visible" : "none");
    }
  }

  function updateNote() {
    var n = $("map-note");
    var bits = [];
    if (map.getZoom() < M.zoom.tract_min) bits.push(M.copy.county_chip);
    else if (tractOnly()) bits.push("Census tracts — " + crimeDef(state.crime).label.toLowerCase() + " is not published below tract level");
    else if (map.getZoom() < M.zoom.bg_min) bits.push("Census tracts");
    if (!bits.length) { n.hidden = true; return; }
    n.hidden = false;
    n.textContent = bits[0];
  }

  function setCrime(key) {
    state.crime = key;
    renderChoices();
    repaint();
    refreshSelection();
    writeUrl();
  }

  function setMeasure(key) {
    state.measure = key;
    renderChoices();
    repaint();
    refreshSelection();
    writeUrl();
  }

  /* --------------------------------------------------------------- card -- */

  /* Share of the displayed estimate's expected count that comes from direct
     incident records. Composites carry a precomputed share in the shard; a
     single offence uses its own source mode. Reliability tier and the p10/p90
     range are not published in 2025.1: the tier is "low" for 98.6% of cells,
     including direct-feed cells, and held-out interval coverage is below
     nominal. */
  function directShare(rec) {
    if (!rec) return null;
    var ci = M.composite_index[state.crime];
    if (ci !== undefined) {
      var arr = rec[9];
      if (!arr || arr[ci] === null || arr[ci] === undefined) return null;
      return arr[ci];
    }
    var oi = M.offense_index[state.crime];
    if (oi === undefined || !rec[8]) return null;
    var w = M.mode_direct_weight[String(rec[8][oi])];
    return w === undefined ? null : w;
  }

  function srcPhrase(rec) {
    var share = directShare(rec);
    if (share === null) return "";
    var t = M.direct_share_threshold;
    if (share >= t) return M.copy.source_direct;
    if (1 - share >= t) return M.copy.source_modeled;
    return share >= 0.5 ? M.copy.source_mostly_direct : M.copy.source_mostly_modeled;
  }

  /* The denominator the displayed number is divided by, in words. Every exposure
     denominator is modeled, so the phrase says so; a composite divides each offence
     by that offence's own base and has no single denominator of its own. */
  function denomPhrase() {
    if (state.measure === "resident") return M.copy.denominator_resident;
    var d = M.denominator_exposure[state.crime];
    return d || M.copy.denominator_exposure_composite;
  }

  /* Whether the agency total this neighbourhood's share was cut from was filed in
     full for the data year or estimated. Different question from the source
     phrase, which is about how that total was spread WITHIN the jurisdiction: a cell
     can be modelled from a reported total, or allocated from an estimated one. */
  function totalPhrase(rec, level) {
    if (level === "county") return "";
    var arr = rec && rec[10];
    if (!arr) return "";
    var oi = M.offense_index[state.crime];
    var members = oi !== undefined ? [oi] : M.composite_members[state.crime];
    var seen = false, anyReported = false, anyEstimated = false;
    members.forEach(function (i) {
      var v = arr[i];
      if (v === null || v === undefined) return;
      seen = true;
      if (v === 1) anyReported = true; else anyEstimated = true;
    });
    if (!seen) return "";
    if (!anyEstimated) return M.copy.total_reported;
    if (!anyReported) return M.copy.total_estimated;
    return M.copy.total_mixed;
  }

  /* Suppressed area: the expected annual count, named by the area type. */
  function suppressedCountText(rec, ec) {
    var sc = M.suppressed_count;
    var noun = M.count_noun[state.crime] || ["offense", "offenses"];
    var label = rec && rec[5] ? M.special_use[rec[5]] : sc.ordinary_label;
    var n = Math.round(ec);
    var amount = ec < 0.5 ? "fewer than one " + noun[0] : "about " + fmt(n) + " " + (n === 1 ? noun[0] : noun[1]);
    var tail = state.measure === "resident" ? sc.resident_tail : sc.exposure_tail;
    return label + ": " + amount + " a year; " + tail;
  }

  function expectedCount(rec) {
    if (!rec || !rec[7]) return null;
    var idx = M.offense_index[state.crime];
    var members = idx !== undefined ? [idx] : M.composite_members[state.crime];
    var total = 0, seen = false;
    members.forEach(function (i) {
      var v = rec[7][i];
      if (v !== null && v !== undefined) { total += v; seen = true; }
    });
    return seen ? total : null;
  }

  function placeLine(rec, level, geoid) {
    var bits = [];
    if (rec && rec[0]) bits.push(rec[0]);
    var tail = [];
    if (rec && rec[1]) tail.push(rec[1]);
    var st = geoid.slice(0, 2);
    var stname = M.state_name_by_fips[st];
    if (stname) tail.push(stname);
    if (level === "county") return (rec && rec[0] ? rec[0] : geoid) + (stname ? ", " + stname : "");
    return bits.concat(tail.length ? [tail.join(", ")] : []).join(" · ");
  }

  function idLine(level, geoid) {
    if (level === "county") return "County " + geoid;
    if (level === "tract") return "Census tract " + geoid;
    return "Census block group " + geoid;
  }

  /* Resolves a selection to what the card shows:
       {rec, value, status, level, geoid, requested, note, error}
     A block group never silently becomes its tract. Murder and rape at block
     group are shown at the tract with a note saying so (the map itself shows
     tracts for them); a suppressed or empty block group shows its own status and
     offers the tract as an explicit action (state.tractView). */
  function resolve(ref) {
    var level = ref.level;
    var geoid = ref.geoid;
    var note = null;
    if (level === "bg" && tractOnly()) {
      level = "tract"; geoid = geoid.slice(0, 11); note = "tract_only";
    } else if (level === "bg" && state.tractView === ref.geoid) {
      level = "tract"; geoid = geoid.slice(0, 11); note = "tract_estimate";
    }
    var mi = measureIndex();
    return lookup(level, geoid).then(function (rec) {
      var v = rec && rec[6] ? rec[6][mi] : null;
      if (v === undefined) v = null;
      return {
        rec: rec, value: v, status: recStatus(rec, mi), level: level, geoid: geoid,
        requested: ref, note: note, error: false
      };
    }, function () {
      return {
        rec: null, value: null, status: null, level: level, geoid: geoid,
        requested: ref, note: note, error: true
      };
    });
  }

  function badge(text) {
    var b = document.createElement("span");
    b.className = "badge";
    b.textContent = text;
    return b;
  }

  function actionButton(text, onClick) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  function renderCard(r) {
    var card = $("card");
    var cls = legendClass(r.value);
    var rec = r.rec;
    var c = M.status.card;
    var published = r.value !== null && r.value !== undefined;
    var frag = document.createElement("div");

    var close = document.createElement("button");
    close.className = "close";
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    close.innerHTML = "&times;";
    close.addEventListener("click", clearSelection);

    var place = document.createElement("p");
    place.className = "place";
    place.textContent = placeLine(rec, r.level, r.geoid);
    if (rec && rec[5]) {
      place.appendChild(document.createTextNode(" "));
      place.appendChild(badge(M.special_use[rec[5]]));
    }
    if (r.note === "tract_estimate") {
      place.appendChild(document.createTextNode(" "));
      place.appendChild(badge(c.tract_estimate_badge));
    }

    var head = document.createElement("p");
    head.className = "headline";
    head.innerHTML = measureLabel().replace(/^./, function (m) { return m.toUpperCase(); }) +
      ": <span class=\"n\">" + fmt(r.value) +
      "</span>" + (cls ? " — " + cls.label : "");

    var times = document.createElement("p");
    times.className = published ? "times" : "times status";
    times.textContent = published
      ? (r.value / 100).toFixed(r.value < 1000 ? 1 : 0) + "× U.S. average"
      : statusText(r);

    var noteRow = document.createElement("p");
    noteRow.className = "row note";
    noteRow.textContent = r.note === "tract_only" ? tractOnlyText() : "";

    var counts = document.createElement("p");
    counts.className = "row";
    var ec = expectedCount(rec);
    // Whole offences. A tenth of an offence is not a thing the model knows. The
    // count is published even where the index is suppressed.
    counts.textContent = ec === null
      ? ""
      : (!published && !r.error && r.status === ST.SUP && r.level !== "county")
        ? suppressedCountText(rec, ec)
        : "Expected offenses per year: " + (ec < 0.5 ? "under 1" : fmt(ec));

    // Both measures on the card: the other measure's value for the same crime and area.
    var otherKey = state.measure === "exposure" ? "resident" : "exposure";
    var otherField = (otherKey === "exposure" ? "x_" : "r_") + state.crime;
    var oi = M.measure_order.indexOf(otherField);
    var ov = rec && rec[6] ? rec[6][oi] : null;
    var other = document.createElement("p");
    other.className = "row";
    var otherLabel = M.measures.filter(function (m) { return m.key === otherKey; })[0].label;
    other.textContent = r.error || r.level === "county" ? "" : otherLabel + ": " +
      (ov === null || ov === undefined ? statusShort({ status: recStatus(rec, oi) }) : fmt(ov));

    var help = document.createElement("p");
    help.className = "row note";
    help.textContent = helpText(state.crime, state.measure);

    var denom = document.createElement("p");
    denom.className = "row";
    denom.textContent = published ? denomPhrase() : "";

    var total = document.createElement("p");
    total.className = "row";
    total.textContent = r.error ? "" : totalPhrase(rec, r.level);

    var benchRow = document.createElement("p");
    benchRow.className = "row";
    benchRow.textContent = "";

    var gid = document.createElement("p");
    gid.className = "geoid";
    gid.textContent = idLine(r.level, r.geoid);

    var src = document.createElement("p");
    src.className = "source";
    src.textContent = r.error ? "" : (r.level === "county" ? M.copy.county_note : srcPhrase(rec));

    // Explicit actions: retry a failed load; offer the parent tract for a block
    // group that publishes no value; return from the tract to the block group.
    var acts = document.createElement("p");
    acts.className = "actions";
    if (r.error) {
      acts.appendChild(actionButton(c.retry, function () { refreshSelection(); }));
    } else if (r.requested.level === "bg" && r.level === "bg" && !published) {
      acts.appendChild(actionButton(c.show_tract, function () {
        state.tractView = r.requested.geoid;
        applySelectionFilters();
        refreshSelection();
      }));
    } else if (r.note === "tract_estimate") {
      acts.appendChild(actionButton(c.back_to_block_group, function () {
        state.tractView = null;
        applySelectionFilters();
        refreshSelection();
      }));
    }

    var links = document.createElement("p");
    links.className = "links";
    var cmpBtn = document.createElement("button");
    cmpBtn.type = "button";
    cmpBtn.textContent = "Compare";
    cmpBtn.addEventListener("click", startCompare);
    links.appendChild(cmpBtn);
    links.appendChild(dot());
    links.appendChild(link("Methods", "methods.html"));
    links.appendChild(dot());
    links.appendChild(link("Download", "download.html"));

    frag.appendChild(close);
    frag.appendChild(place);
    frag.appendChild(head);
    frag.appendChild(times);
    if (noteRow.textContent) frag.appendChild(noteRow);
    if (acts.childNodes.length) frag.appendChild(acts);
    if (other.textContent) frag.appendChild(other);
    if (counts.textContent) frag.appendChild(counts);
    if (denom.textContent) frag.appendChild(denom);
    if (total.textContent) frag.appendChild(total);
    frag.appendChild(benchRow);
    frag.appendChild(gid);
    if (src.textContent) frag.appendChild(src);
    frag.appendChild(help);
    frag.appendChild(links);

    card.innerHTML = "";
    card.appendChild(frag);
    card.classList.add("on");
    card.setAttribute("data-status", r.error ? "error" : String(r.status));
    syncSheet();

    var st = M.state_abbr_by_fips[r.geoid.slice(0, 2)];
    if (st && !r.error) {
      bench(st).then(function (bd) {
        if (!bd) return;
        var mi = bd.measures.indexOf(field());
        var parts = [];
        var jid = rec ? rec[2] : "";
        // A jurisdiction below the support floor has no benchmark worth
        // printing; the line falls back to state and national.
        if (jid && bd.j[jid] && bd.j[jid].s !== 0) {
          parts.push(bd.j[jid].name + " " + fmt(bd.j[jid].v[mi]));
        }
        parts.push(bd.state_name + " " + fmt(bd.state[field()]));
        parts.push("U.S. 100");
        benchRow.textContent = parts.join(" · ");
      });
    }
  }

  function dot() {
    var s = document.createElement("span");
    s.className = "dot";
    s.textContent = "·";
    return s;
  }

  function link(text, href) {
    var a = document.createElement("a");
    a.textContent = text;
    a.href = href;
    return a;
  }

  /* The outline follows what the card shows: the block group, or its tract when
     the tract estimate was asked for. */
  function shownRef() {
    if (!state.sel) return null;
    if (state.sel.level === "bg" && (tractOnly() || state.tractView === state.sel.geoid)) {
      return { level: "tract", geoid: state.sel.geoid.slice(0, 11) };
    }
    return state.sel;
  }

  function applySelectionFilters() {
    var shown = shownRef();
    ["county", "tract", "bg"].forEach(function (level) {
      var gid = shown && shown.level === level ? shown.geoid : "";
      map.setFilter("selhalo-" + level, ["==", ["get", "geoid"], gid]);
      map.setFilter("sel-" + level, ["==", ["get", "geoid"], gid]);
    });
  }

  function select(level, geoid, opts) {
    state.sel = { level: level, geoid: geoid };
    state.tractView = null;
    applySelectionFilters();
    writeUrl();
    return resolve(state.sel).then(function (r) {
      if (state.cmp) { renderCompare(); } else { renderCard(r); }
      setSnap("detail");
      if (opts && opts.focus) $("card").focus();
      if (!opts || opts.fit !== false) fitShown();
      return r;
    });
  }

  function refreshSelection() {
    if (state.cmp) { renderCompare(); return; }
    if (!state.sel) return;
    resolve(state.sel).then(function (r) { renderCard(r); fitShown(); });
  }

  function clearSelection() {
    state.sel = null;
    state.cmp = null;
    state.tractView = null;
    applySelectionFilters();
    $("card").classList.remove("on");
    $("compare").classList.remove("on");
    if (state.marker) { state.marker.remove(); state.marker = null; }
    setSnap("collapsed");
    writeUrl();
  }

  /* ---------------------------------------------------- fit to viewport -- */

  /* The part of the map not covered by the card (desktop) or the bottom sheet
     (mobile), as padding in CSS pixels. */
  function obscuredPadding() {
    var pad = { top: 16, right: 16, bottom: 16, left: 16 };
    var wrap = document.querySelector(".map-wrap").getBoundingClientRect();
    if (isMobile()) {
      var sheet = $("sidebar").getBoundingClientRect();
      pad.bottom += Math.max(0, wrap.bottom - sheet.top);
      pad.top += 44;                         // map note / attribution row
    } else {
      ["card", "compare"].forEach(function (id) {
        var el = $(id);
        if (!el.classList.contains("on")) return;
        var b = el.getBoundingClientRect();
        pad.right = Math.max(pad.right, wrap.right - b.left + 12);
      });
    }
    return pad;
  }

  /* Bounding box of every rendered piece of a feature (tiles clip polygons, so
     one area can arrive as several parts). */
  function featureBounds(level, geoid) {
    var feats = map.querySourceFeatures(SRC[level], {
      sourceLayer: M.tiles[level].layer,
      filter: ["==", ["get", "geoid"], geoid]
    });
    if (!feats.length) return null;
    var b = [Infinity, Infinity, -Infinity, -Infinity];
    feats.forEach(function (f) {
      var g = f.geometry;
      var polys = g.type === "Polygon" ? [g.coordinates] : (g.type === "MultiPolygon" ? g.coordinates : []);
      polys.forEach(function (poly) {
        poly[0].forEach(function (pt) {
          if (pt[0] < b[0]) b[0] = pt[0];
          if (pt[1] < b[1]) b[1] = pt[1];
          if (pt[0] > b[2]) b[2] = pt[0];
          if (pt[1] > b[3]) b[3] = pt[1];
        });
      });
    });
    return isFinite(b[0]) ? [[b[0], b[1]], [b[2], b[3]]] : null;
  }

  /* Keep the shown area inside the unobscured part of the map. It pans, and
     zooms out only when the area is larger than the free space; it never zooms
     in past the reader's own zoom. */
  function fitShown(retried) {
    var ref = shownRef();
    if (!ref || !map) return;
    requestAnimationFrame(function () {
      var bb = featureBounds(ref.level, ref.geoid);
      // A selection restored from a URL can arrive before its tiles: try once more
      // when the map next settles.
      if (!bb) {
        if (!retried) { map.once("idle", function () { fitShown(true); }); map.triggerRepaint(); }
        return;
      }
      var pad = obscuredPadding();
      var canvas = map.getCanvas();
      var w = canvas.clientWidth, h = canvas.clientHeight;
      var sw = map.project(bb[0]), ne = map.project(bb[1]);
      var inside = sw.x >= pad.left && ne.x <= w - pad.right &&
        ne.y >= pad.top && sw.y <= h - pad.bottom;
      if (inside) return;
      map.fitBounds(bb, { padding: pad, maxZoom: map.getZoom(), duration: 450 });
    });
  }

  /* ------------------------------------------------------------ compare -- */

  var awaitingCompare = false;

  function startCompare() {
    if (!state.sel) return;
    awaitingCompare = true;
    $("card").classList.remove("on");
    var c = $("compare");
    c.classList.add("on");
    c.innerHTML = "<p class=\"hint\">Search for a second place, or click another area on the map.</p>";
    var close = document.createElement("button");
    close.className = "close";
    close.type = "button";
    close.setAttribute("aria-label", "Close comparison");
    close.innerHTML = "&times;";
    close.addEventListener("click", function () { endCompare(); });
    c.appendChild(close);
    setSnap("detail");
    c.focus();
  }

  function endCompare() {
    awaitingCompare = false;
    state.cmp = null;
    $("compare").classList.remove("on");
    writeUrl();
    refreshSelection();
    if (state.sel) $("card").classList.add("on");
  }

  function cellText(v) { return typeof v === "string" ? v : fmt(v); }

  function renderCompare() {
    if (!state.sel || !state.cmp) return;
    var c = $("compare");
    Promise.all([resolve(state.sel), resolve(state.cmp)]).then(function (rs) {
      var a = rs[0], b = rs[1];
      var stA = M.state_abbr_by_fips[a.geoid.slice(0, 2)];
      var stB = M.state_abbr_by_fips[b.geoid.slice(0, 2)];
      Promise.all([bench(stA), bench(stB)]).then(function (bds) {
        c.innerHTML = "";
        var close = document.createElement("button");
        close.className = "close";
        close.type = "button";
        close.setAttribute("aria-label", "Close comparison");
        close.innerHTML = "&times;";
        close.addEventListener("click", endCompare);
        c.appendChild(close);

        var h = document.createElement("h2");
        h.textContent = measureLabel().replace(/^./, function (m) { return m.toUpperCase(); });
        c.appendChild(h);

        var f = field();
        function row(label, va, vb, cls) {
          return { label: label, a: va, b: vb, cls: cls || "" };
        }
        function jname(bd, rec) {
          var jid = rec ? rec[2] : "";
          return bd && jid && bd.j[jid] ? bd.j[jid].name : "Police jurisdiction";
        }
        function jval(bd, rec) {
          var jid = rec ? rec[2] : "";
          if (!bd || !jid || !bd.j[jid] || bd.j[jid].s === 0) return null;
          return bd.j[jid].v[bd.measures.indexOf(f)];
        }

        var ja = jval(bds[0], a.rec), jb = jval(bds[1], b.rec);
        var rows = [row("This area",
          a.value === null ? statusShort(a) : a.value,
          b.value === null ? statusShort(b) : b.value)];
        if (ja !== null || jb !== null) rows.push(row("Police jurisdiction", ja, jb));
        rows.push(
          row("State", bds[0] ? bds[0].state[f] : null, bds[1] ? bds[1].state[f] : null),
          row("United States", 100, 100)
        );

        var table = document.createElement("table");
        var thead = document.createElement("thead");
        var htr = document.createElement("tr");
        [
          "",
          placeLine(a.rec, a.level, a.geoid),
          placeLine(b.rec, b.level, b.geoid)
        ].forEach(function (t) {
          var th = document.createElement("th");
          th.textContent = t;
          htr.appendChild(th);
        });
        thead.appendChild(htr);
        table.appendChild(thead);

        var tbody = document.createElement("tbody");
        rows.forEach(function (r) {
          var tr = document.createElement("tr");
          var lab = r.label;
          if (r.label === "Police jurisdiction") {
            var na = ja === null ? null : jname(bds[0], a.rec);
            var nb = jb === null ? null : jname(bds[1], b.rec);
            lab = na === nb ? (na || "Police jurisdiction")
                : [na, nb].filter(Boolean).join(" / ");
          }
          if (r.label === "State") {
            var sa = bds[0] ? bds[0].state_name : "State";
            var sb = bds[1] ? bds[1].state_name : "State";
            lab = sa === sb ? sa : sa + " / " + sb;
          }
          [lab, cellText(r.a), cellText(r.b)].forEach(function (t) {
            var td = document.createElement("td");
            td.textContent = t;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });

        var d = document.createElement("tr");
        d.className = "diff";
        var diff = (a.value !== null && b.value !== null) ? (b.value - a.value) : null;
        var pct = (diff !== null && a.value) ? (100 * diff / a.value) : null;
        var cells = [
          "Difference",
          "",
          diff === null ? "—"
            : (diff > 0 ? "+" : "") + fmt(diff) + (pct === null ? "" : " (" + (pct > 0 ? "+" : "") + Math.round(pct) + "%)")
        ];
        cells.forEach(function (t) {
          var td = document.createElement("td");
          td.textContent = t;
          d.appendChild(td);
        });
        tbody.appendChild(d);
        table.appendChild(tbody);
        c.appendChild(table);

        var p = document.createElement("p");
        p.className = "links";
        p.appendChild(link("Methods", "methods.html"));
        p.appendChild(dot());
        p.appendChild(link("Download", "download.html"));
        c.appendChild(p);
        c.classList.add("on");
        syncSheet();
      });
    });
  }

  /* ------------------------------------------------------------- search -- */

  var lastRequest = 0;

  /* The published map is the 48 contiguous states and DC. Anything outside that
     box is a real place the map has no data for, which is a different answer from
     "no such place" and from "the geocoder did not respond". */
  function inCoverage(item) {
    var b = M.coverage_bounds;
    var lng = parseFloat(item.lon), lat = parseFloat(item.lat);
    if (isNaN(lng) || isNaN(lat)) return false;
    return lng >= b[0] && lng <= b[2] && lat >= b[1] && lat <= b[3];
  }

  /* Resolves to {ok, items}. `ok:false` means the geocoding service failed; it is
     never reported as a bad address. The viewbox biases the service towards the
     published area; the coverage test above is what actually restricts it. */
  function geocode(q) {
    var wait = Math.max(0, 1000 - (Date.now() - lastRequest));
    return new Promise(function (res) { setTimeout(res, wait); }).then(function () {
      lastRequest = Date.now();
      var b = M.coverage_bounds;
      var url = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&countrycodes=us" +
        "&viewbox=" + [b[0], b[3], b[2], b[1]].join(",") +
        "&q=" + encodeURIComponent(q);
      return fetch(url, { headers: { "Accept": "application/json" } })
        .then(function (r) {
          if (!r.ok) throw new Error("geocoder http " + r.status);
          return r.json();
        })
        .then(function (items) {
          return { ok: true, items: Array.isArray(items) ? items : [] };
        })
        .catch(function () { return { ok: false, items: [] }; });
    });
  }

  function renderResults(items) {
    var ul = $("results");
    ul.innerHTML = "";
    $("q").setAttribute("aria-expanded", String(items.length > 0));
    items.forEach(function (it, i) {
      var li = document.createElement("li");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", "false");
      li.id = "opt-" + i;
      li.tabIndex = -1;
      li.textContent = it.display_name;
      li.addEventListener("click", function () { pick(it); });
      ul.appendChild(li);
    });
  }

  function pick(item) {
    if (!inCoverage(item)) {
      $("search-note").textContent = M.copy.search_outside_coverage;
      return;
    }
    $("results").innerHTML = "";
    $("q").setAttribute("aria-expanded", "false");
    $("q").value = item.display_name.split(",").slice(0, 2).join(",").trim();
    var lng = parseFloat(item.lon), lat = parseFloat(item.lat);
    if (state.marker) state.marker.remove();
    state.marker = new maplibregl.Marker({ color: "#16181c" }).setLngLat([lng, lat]).addTo(map);
    map.easeTo({ center: [lng, lat], zoom: Math.max(map.getZoom(), 12), duration: 600 });
    map.once("moveend", function () {
      whenReady(function () { selectAtPoint([lng, lat]); });
    });
  }

  function whenReady(cb) {
    if (map.areTilesLoaded() && map.isStyleLoaded()) { cb(); return; }
    var tries = 0;
    var tick = function () {
      if ((map.areTilesLoaded() && map.isStyleLoaded()) || ++tries > 60) cb();
      else setTimeout(tick, 120);
    };
    setTimeout(tick, 120);
  }

  function selectAtPoint(lngLat) {
    var layer = activeFillLayer();
    var pt = map.project(lngLat);
    var hits = map.queryRenderedFeatures(pt, { layers: [layer] });
    if (!hits.length && layer === "fill-bg") {
      hits = map.queryRenderedFeatures(pt, { layers: ["fill-tract"] });
    }
    if (!hits.length) {
      $("search-note").textContent = "No published area at that point.";
      return;
    }
    $("search-note").textContent = "";
    var level = hits[0].layer.id === "fill-bg" ? "bg" : (hits[0].layer.id === "fill-tract" ? "tract" : "county");
    var gid = hits[0].properties.geoid;
    if (awaitingCompare) {
      awaitingCompare = false;
      state.cmp = { level: level, geoid: gid };
      writeUrl();
      renderCompare();
      setSnap("detail");
    } else {
      // Focus follows the result so a keyboard user lands on the card.
      select(level, gid, { focus: true });
    }
  }

  function wireSearch() {
    var form = $("search-form"), input = $("q"), ul = $("results");
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var q = input.value.trim();
      if (!q) return;
      $("search-note").textContent = "Searching…";
      geocode(q).then(function (res) {
        if (!res.ok) {
          $("search-note").textContent = M.copy.search_unavailable;
          renderResults([]);
          return;
        }
        var inside = res.items.filter(inCoverage);
        if (inside.length) {
          $("search-note").textContent = "";
        } else if (res.items.length) {
          $("search-note").textContent = M.copy.search_outside_coverage;
        } else {
          $("search-note").textContent = M.copy.search_no_match;
        }
        renderResults(inside);
        if (inside.length === 1) pick(inside[0]);
      });
    });

    var active = -1;
    input.addEventListener("keydown", function (e) {
      var opts = ul.querySelectorAll("li");
      if (!opts.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        active = e.key === "ArrowDown"
          ? Math.min(active + 1, opts.length - 1)
          : Math.max(active - 1, 0);
        opts.forEach(function (o, i) { o.setAttribute("aria-selected", String(i === active)); });
        input.setAttribute("aria-activedescendant", opts[active].id);
        opts[active].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && active >= 0) {
        e.preventDefault();
        opts[active].click();
        active = -1;
      } else if (e.key === "Escape") {
        ul.innerHTML = "";
        input.setAttribute("aria-expanded", "false");
        active = -1;
      }
    });
  }

  /* ------------------------------------------------------ mobile sheet -- */

  /* On a phone the sidebar is ONE bottom sheet with three snap states:
     collapsed (title, active measure and a legend strip), controls (search,
     pickers, full legend) and detail (the selected area's card or the
     comparison). The card and the comparison move into the sheet; on a wide
     screen they float over the map as before. */
  var MOBILE = window.matchMedia("(max-width: 700px)");

  function isMobile() { return MOBILE.matches; }

  function placePanels() {
    var host = isMobile() ? $("sheet-detail") : document.querySelector(".map-wrap");
    ["card", "compare"].forEach(function (id) {
      if ($(id).parentNode !== host) host.appendChild($(id));
    });
    syncSheet();
  }

  function setSnap(snap) {
    var open = $("card").classList.contains("on") || $("compare").classList.contains("on");
    if (snap === "detail" && !open) snap = "collapsed";
    $("sidebar").setAttribute("data-snap", snap);
    Array.prototype.forEach.call(document.querySelectorAll("#sheet-tabs button"), function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-snap") === snap));
    });
    syncSheet();
  }

  /* Map controls anchored to the bottom edge sit above the sheet. */
  function syncSheet() {
    var open = $("card").classList.contains("on") || $("compare").classList.contains("on");
    var detailBtn = document.querySelector('#sheet-tabs button[data-snap="detail"]');
    if (detailBtn) detailBtn.disabled = !open;
    var h = isMobile() ? Math.round($("sidebar").getBoundingClientRect().height) : 0;
    document.querySelector(".map-wrap").style.setProperty("--sheet-h", h + "px");
  }

  function renderSheetSummary() {
    var el = $("sheet-summary");
    if (el) el.textContent = measureLabel().replace(/^./, function (m) { return m.toUpperCase(); });
  }

  function wireSheet() {
    Array.prototype.forEach.call(document.querySelectorAll("#sheet-tabs button"), function (b) {
      b.addEventListener("click", function () { setSnap(b.getAttribute("data-snap")); });
    });
    if (MOBILE.addEventListener) MOBILE.addEventListener("change", placePanels);
    else if (MOBILE.addListener) MOBILE.addListener(placePanels);
    if (window.ResizeObserver) new ResizeObserver(syncSheet).observe($("sidebar"));
    placePanels();
    setSnap(isMobile() ? "collapsed" : "controls");
  }

  /* --------------------------------------------------------------- boot -- */

  /* First data paint: the first frame rendered after a choropleth TILE has
     loaded. performance.now() counts from navigation start. Until 2025.1.1 this
     fired on the first `sourcedata` event whose source reported loaded, which is
     the PMTiles header arriving (no tile requested yet); that moment is still
     recorded, as __crimeriskFirstMetadata, so old and new numbers can be compared. */
  var tileLoaded = false;

  function reportMetadata() {
    if (window.__crimeriskFirstMetadata) return;
    window.__crimeriskFirstMetadata = Math.round(performance.now());
  }

  function reportPaint() {
    if (window.__crimeriskFirstPaint || !tileLoaded) return;
    window.__crimeriskFirstPaint = Math.round(performance.now());
    window.__crimeriskScriptStart = Math.round(T0);
    document.documentElement.setAttribute("data-first-data-paint", window.__crimeriskFirstPaint);
  }

  function wireMap() {
    var hover = $("hover"), hoverGeoid = null;

    map.on("mousemove", function (e) {
      var layer = activeFillLayer();
      if (!map.getLayer(layer)) return;
      var hits = map.queryRenderedFeatures(e.point, { layers: [layer] });
      var level = activeLevel();
      if (!hits.length) {
        hover.style.display = "none";
        hoverGeoid = null;
        ["county", "tract", "bg"].forEach(function (l) {
          map.setFilter("hover-" + l, ["==", ["get", "geoid"], ""]);
        });
        return;
      }
      var p = hits[0].properties;
      var gid = p.geoid;
      var v = p[field()];
      map.setFilter("hover-" + level, ["==", ["get", "geoid"], gid]);
      var cls = legendClass(v);
      var shown;
      if (v === undefined || v === null) {
        var st = tileStatus(p, measureIndex());
        shown = st === ST.SUP ? "Not published" : "No estimate";
      } else {
        shown = fmt(v);
      }
      hover.style.display = "block";
      hover.style.left = Math.min(e.point.x + 14, map.getCanvas().clientWidth - 270) + "px";
      hover.style.top = (e.point.y + 14) + "px";
      hover.innerHTML = "<span class=\"v\"><b>" + shown + "</b>" +
        (cls ? " — " + cls.label : "") + "</span>";
      hoverGeoid = gid;
      lookup(level, gid).then(function (rec) {
        if (hoverGeoid !== gid || !rec) return;
        var nm = document.createElement("div");
        nm.textContent = placeLine(rec, level, gid);
        hover.insertBefore(nm, hover.firstChild);
      }, function () { /* the card reports load errors; the hover chip stays short */ });
    });

    map.on("mouseout", function () { hover.style.display = "none"; });

    map.on("click", function (e) {
      var layer = activeFillLayer();
      var hits = map.queryRenderedFeatures(e.point, { layers: [layer] });
      if (!hits.length) return;
      var level = activeLevel();
      if (awaitingCompare) {
        awaitingCompare = false;
        state.cmp = { level: level, geoid: hits[0].properties.geoid };
        writeUrl();
        renderCompare();
        setSnap("detail");
      } else {
        select(level, hits[0].properties.geoid, { focus: false });
      }
    });

    map.on("zoomend", function () { updateNote(); writeUrl(); });
    map.on("moveend", writeUrl);
    map.on("sourcedata", function (e) {
      if (!e.sourceId || ["c", "t", "b"].indexOf(e.sourceId) < 0) return;
      if (e.isSourceLoaded) reportMetadata();
      if (e.tile) tileLoaded = true;
    });
    map.on("render", reportPaint);
  }

  function boot(manifest) {
    M = manifest;
    var view = readUrl();

    renderChoices();
    renderLegend();
    $("edition").textContent = M.copy.edition_line;
    document.title = "CrimeRisk — " + M.copy.tagline;

    renderLegendStrip();
    renderSheetSummary();
    wireSheet();

    // The archives' headers are requested as soon as the manifest names them, in
    // parallel with the map's own start-up, instead of after the map's load event.
    var proto = new pmtiles.Protocol();
    maplibregl.addProtocol("pmtiles", proto.tile);
    ["county", "tract", "bg"].forEach(function (level) {
      var archive = new pmtiles.PMTiles(M.tiles[level].url);
      proto.add(archive);
      archive.getHeader().catch(function () { /* the map reports tile errors */ });
    });

    // No sprite in the initial style: the map's load event waits for the sprite,
    // and the choropleth sources are only added on load, so a third-party sprite
    // host sat on the critical path of the first data paint. The basemap's sprite
    // is set when the basemap itself is merged.
    map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        glyphs: M.basemap.glyphs,
        sources: {},
        layers: [{ id: "paper", type: "background", paint: { "background-color": "#f4f2ee" } }]
      },
      bounds: M.bounds,
      fitBoundsOptions: { padding: 18 },
      maxZoom: 16,
      minZoom: 3,
      attributionControl: { compact: isMobile() }
    });
    if (view) map.jumpTo({ center: view.center, zoom: view.zoom });

    window.__crimeriskMap = map;   // inspection hook for the verification step
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "imperial" }), "bottom-left");

    map.on("load", function () {
      try {
        map.addImage("hatch", hatchImage(false), { pixelRatio: 2 });
        map.addImage("hatch-sup", hatchImage(true, "rgba(70,72,76,0.55)"), { pixelRatio: 2 });
      } catch (e) { /* pattern is optional */ }
      addChoropleth();
      applySelectionFilters();
      repaint();
      wireMap();

      fetch(M.basemap.style)
        .then(function (r) { return r.json(); })
        .then(mergeBasemap)
        .catch(function () { /* choropleth stands on its own */ });

      Promise.all([loadShardIndex("bg"), loadShardIndex("tract"), loadShardIndex("county")])
        .then(function () {
          if (state.sel) select(state.sel.level, state.sel.geoid, { focus: false });
          if (state.cmp) renderCompare();
        });
    });

    map.on("error", function (e) {
      if (e && e.error && /pmtiles|Failed to fetch/i.test(String(e.error))) {
        $("map-error").classList.add("on");
      }
    });

    wireSearch();
    $("special-toggle").addEventListener("change", function () {
      updateHatch();
      renderLegend();
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        if ($("compare").classList.contains("on")) endCompare();
        else if ($("card").classList.contains("on")) clearSelection();
      }
    });
  }

  fetch("data/manifest.json")
    .then(function (r) { return r.json(); })
    .then(boot)
    .catch(function () { $("map-error").classList.add("on"); });
})();
