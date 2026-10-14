const $ = (id) => document.getElementById(id);
// No pre-selected location: the user picks one via search or GPS first.
const state = { lat: null, lon: null, place: null };
let chart = null;

// "Gachibowli, Greater Hyderabad Municipal Corporation West Zone, ..." -> "Gachibowli, Hyderabad"
function shortName(name) {
  const parts = String(name || "").split(",").map((s) => s.trim()).filter(Boolean);
  return parts.slice(0, 2).join(", ") || "Selected location";
}

async function readJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setPlace(lat, lon, place) {
  state.lat = Number(lat);
  state.lon = Number(lon);
  state.place = place || "Selected location";
  const nameEl = $("place-name");
  nameEl.textContent = state.place;
  nameEl.classList.remove("muted");
  $("place-coords").textContent = `(${state.lat.toFixed(5)}, ${state.lon.toFixed(5)})`;
  $("search-results").replaceChildren();
  fetchWeather();
}

$("search-btn").addEventListener("click", async () => {
  const query = $("search-input").value.trim();
  if (!query) return;
  $("search-results").textContent = "Searching locations…";
  try {
    const results = await readJson(`/api/geocode?q=${encodeURIComponent(query)}`);
    $("search-results").replaceChildren();
    if (!Array.isArray(results) || results.length === 0) {
      $("search-results").textContent = "No places found. Try a locality plus city, e.g. Gachibowli, Hyderabad.";
      return;
    }
    results.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "result-btn";
      button.textContent = item.name.length > 90 ? item.name.slice(0, 90) + "…" : item.name;
      button.addEventListener("click", () => setPlace(item.lat, item.lon, shortName(item.name)));
      $("search-results").appendChild(button);
    });
  } catch (error) {
    $("search-results").textContent = `${error.message}. Check your internet connection and restart Flask.`;
  }
});

$("search-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    $("search-btn").click();
  }
});

$("geo-btn").addEventListener("click", () => {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by this browser. Search for your place instead.");
    return;
  }
  $("geo-btn").disabled = true;
  $("geo-btn").textContent = "Locating…";
  navigator.geolocation.getCurrentPosition(async (position) => {
    try {
      const { latitude, longitude } = position.coords;
      let place = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
      try {
        const reverse = await readJson(`/api/reverse?lat=${latitude}&lon=${longitude}`);
        place = reverse.place || place;
      } catch (_) {}
      setPlace(latitude, longitude, place);
    } finally {
      $("geo-btn").disabled = false;
      $("geo-btn").textContent = "Use my location";
    }
  }, (error) => {
    const messages = {
      1: "Location permission was denied. Allow location access in the browser, or search manually.",
      2: "Your device could not determine the location. Search manually.",
      3: "Location request timed out. Try again or search manually."
    };
    alert(messages[error.code] || "Could not get your location. Search manually.");
    $("geo-btn").disabled = false;
    $("geo-btn").textContent = "Use my location";
  }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 });
});

async function fetchWeather() {
  $("wx-cond").textContent = "Loading live weather…";
  try {
    const weather = await readJson(`/api/weather?lat=${state.lat}&lon=${state.lon}`);
    if (!weather.live) throw new Error(weather.error || "Weather unavailable");
    $("wx-temp").textContent = weather.temp_c ?? "—";
    $("wx-cond").textContent = weather.name || "Current conditions";
    $("wx-rain").textContent = `Rain: ${weather.rain_mm ?? 0} mm`;
    $("wx-humidity").textContent = `Humidity: ${weather.humidity ?? "—"}%`;
    $("wx-wind").textContent = `Wind: ${weather.wind_kmh ?? "—"} km/h`;
    $("weather-error").textContent = "";
  } catch (error) {
    $("wx-cond").textContent = "Weather unavailable";
    $("weather-error").textContent = error.message;
  }
}

function trafficClass(level) {
  return String(level || "").toLowerCase();
}

$("festival-check").addEventListener("change", () => {
  $("festival-name").classList.toggle("hidden", !$("festival-check").checked);
  if ($("festival-check").checked) $("holiday-check").checked = true;
});

$("predict-btn").addEventListener("click", async () => {
  const button = $("predict-btn");
  $("error-message").classList.add("hidden");
  if (state.lat == null || state.lon == null) {
    $("error-message").textContent = "Pick a location first — search for a place or tap “Use my location”.";
    $("error-message").classList.remove("hidden");
    return;
  }
  button.disabled = true;
  button.textContent = "Forecasting…";
  try {
    const data = await readJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: state.lat,
        lon: state.lon,
        place: state.place,
        holiday: $("holiday-check").checked,
        festival: $("festival-check").checked,
        festival_name: $("festival-name").value.trim()
      })
    });
    const best = data.best_time;
    $("best-time").textContent = `${best.label} · ${Number(best.volume).toLocaleString("en-IN")} vehicles/hour · ${best.level} traffic`;
    $("source-note").textContent = data.source_note || "";
    const tbody = $("forecast-body");
    tbody.replaceChildren();
    data.rows.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.time === best.time) tr.className = "best-row";
      for (const value of [row.label, Number(row.volume).toLocaleString("en-IN", { maximumFractionDigits: 2 }), row.level]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });
    if (chart) chart.destroy();
    chart = new Chart($("hourly-chart").getContext("2d"), {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [{
          label: "Predicted vehicles/hour",
          data: data.hourly,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37,99,235,0.12)",
          fill: true,
          tension: 0.25,
          pointRadius: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 8, maxRotation: 35 } },
          y: { beginAtZero: true }
        }
      }
    });
    $("result-card").classList.remove("hidden");
    $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
    if (data.weather && data.weather.live) {
      $("wx-temp").textContent = data.weather.temp_c ?? "—";
      $("wx-cond").textContent = data.weather.name || "Current conditions";
    }
  } catch (error) {
    $("error-message").textContent = error.message;
    $("error-message").classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Find best time";
  }
});

// Page starts with no location: weather and prediction wait for the user's pick.
// (fetchWeather() is called from setPlace once a location is chosen.)
$("wx-cond").textContent = "Search for a place or tap “Use my location” to load live weather.";
