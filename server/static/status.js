function formatTimer(secs) {
    const pad = function (n) {
        return (n < 10 ? "0" + n : n);
    };

    let h = Math.floor(secs / 3600);
    let m = Math.floor(secs / 60) - (h * 60);
    let s = Math.floor(secs - h * 3600 - m * 60);

    const strs = [];
    if (h) {
        strs.push(h + 'h');
    }
    if (m || h) {
        if (h) m = pad(m);
        strs.push(m + 'm');
    }
    if (m || h)
        s = pad(s);
    strs.push(s + 's');

    return strs.join(' ');
}

$(window).on('load', function () {
    let updateTimer = null;
    let expectedUpdateInterval = null;
    let updateTimerStart = null;
    const updateTimerEl = $('#update_timer');
    let plotlyConfig = {responsive: true, displayModeBar: false};
    const plotEl = $('#therm_plot');
    const plotDiv = plotEl[0];
    let loadedFrom = null;
    let loadedTo = null;
    let ajaxPending = false;
    let autoRefresh = true;
    const disableAutoRefresh = function () {
        autoRefresh = false;
        if (updateTimer !== null) {
            window.clearTimeout(updateTimer);
            updateTimer = null;
        }
    };
    const adjustTicks = function () {
        const plotWidth = $(plotDiv).width();
        const plotHrRange = ((new Date(plotDiv.layout.xaxis.range[1])).getTime()
                            - (new Date(plotDiv.layout.xaxis.range[0])).getTime()) / 3600000;
        const exp = 1 + Math.floor(Math.log2(36 * (plotHrRange / plotWidth)));
        const scale = (exp > 0) ? (3 * (Math.pow(2, exp) / 2)) : 1;
        const dyntick = scale * 60 * 60 * 1000;
        if (dyntick != plotDiv.layout.xaxis.dtick)
            Plotly.relayout(plotDiv, {"xaxis.dtick": dyntick, "xaxis2.dtick": dyntick, "xaxis3.dtick": dyntick});
    };
    const concatArrays = function (existingArray, newArray, concatMode) {
        if (existingArray === undefined && newArray === undefined)
            return undefined;
        let w0, w1;
        if (concatMode == 'prepend') {
            w0 = newArray; w1 = existingArray;
        }
        if (concatMode == 'extend') {
            w0 = existingArray; w1 = newArray;
        }
        if (w0 === undefined)
            w0 = [];
        if (w1 === undefined)
            w1 = [];
        return w0.concat(w1);
    };
    const recalcAxesRanges = function (plots) {
        for (const key in plotDiv.layout) {
            if (key.indexOf('yaxis') != -1) {
                const existYAxis = plotDiv.layout[key];
                const respYAxis = plots.layout[key];
                if (respYAxis === undefined)
                    continue;
                const existRange = existYAxis.range;
                const respRange = respYAxis.range;
                if (respRange === undefined)
                    continue;
                if (existRange === undefined) {
                    existYAxis.range = respRange;
                } else {
                    const rangeMin = Math.min(existRange[0], respRange[0]);
                    const rangeMax = Math.max(existRange[1], respRange[1]);
                    existYAxis.range = [rangeMin, rangeMax];
                }
            }
        }
    };
    const concatPlotData = function (plots, mode) {
        const plotsData = plots['data'];
        const tracesData = {x: [], y: []};
        const tracesIndices = [];
        const newTraces = [];
        for (const respIdx in plotsData) {  // Add traces missing from existing
            let respTrace = plotsData[respIdx];
            let isNewTrace = true;
            for (const existIdx in plotDiv.data) {
                let existTrace = plotDiv.data[existIdx];
                let respTraceId = respTrace.meta.id;
                let existTraceId = existTrace.meta.id;
                if (respTraceId != existTraceId)
                    continue;
                tracesData.x.push(respTrace.x);
                tracesData.y.push(respTrace.y);
                tracesIndices.push(Number(existIdx));
                if (existTrace.type == 'bar') {
                    existTrace.width = concatArrays(existTrace.width, respTrace.width, mode);
                    existTrace.base = concatArrays(existTrace.base, respTrace.base, mode);
                    existTrace.text = concatArrays(existTrace.text, respTrace.text, mode);
                }
                isNewTrace = false;
                break;
            }
            if (isNewTrace)
                newTraces.push(respTrace);
        }
        plotDiv.layout.shapes = concatArrays(plotDiv.layout.shapes, plots.layout.shapes, mode);
        plotDiv.layout.annotations = concatArrays(plotDiv.layout.annotations, plots.layout.annotations, mode);
        recalcAxesRanges(plots);
        if (tracesIndices.length > 0) {
            if (mode == 'prepend')
                Plotly.prependTraces(plotDiv, tracesData, tracesIndices);
            if (mode == 'extend')
                Plotly.extendTraces(plotDiv, tracesData, tracesIndices);
        }
        if (newTraces.length > 0) {
            Plotly.addTraces(plotDiv, newTraces);
        }
    };
    const plotData = function (plots, mode) {
        Plotly.setPlotConfig(plotlyConfig);
        if (mode == 'prepend' || mode == 'extend') {
            concatPlotData(plots, mode);
        } else {
            Plotly.react(plotDiv, plots);
        }
        adjustTicks();
        // TODO: Reposition last datum annotations based on axis scale to prevent overlap
        // TODO: Automatically move "now" vline regularly and relayout
    };
    const createPlot = function (loadFrom, loadTo, exclusiveBounds, mode) {
        const queryArgs = {};
        if (loadFrom)
            queryArgs['from'] = (new Date(loadFrom)).toISOString();
        if (loadTo)
            queryArgs['to'] = (new Date(loadTo)).toISOString();
        if (exclusiveBounds !== undefined)
            queryArgs['exclusive'] = exclusiveBounds?'1':'0';
        if (mode !== undefined)
            queryArgs['mode'] = mode;
        const loadFail = function () {
            updateTimerEl.text('failed requesting data, try refreshing the page');
            disableAutoRefresh();
        };
        ajaxPending = true;
        $.ajax({
            url: '/plot',
            data: queryArgs,
            cache: false,
            dataType: 'json',
            timeout: 20000
        }).done(function (data /*, textStatus, jqXHR*/) {
            if (data != null && data['status'] == 'SUCCESS') {
                plotData(data['plots'], mode);
                const prependOrExtendMode = (mode == 'prepend' || mode == 'extend');
                const timestampsRange = data['timestamps_range'];
                if (timestampsRange) {
                    if (timestampsRange[0])
                        loadedFrom = prependOrExtendMode ? Math.min(timestampsRange[0], loadedFrom) : timestampsRange[0];
                    if (timestampsRange[1])
                        loadedTo = prependOrExtendMode ? Math.min(timestampsRange[1], loadedTo) : timestampsRange[1];
                }
                expectedUpdateInterval = data['next_update_in'];
                if (autoRefresh && expectedUpdateInterval != null) {
                    updateTimer = window.setTimeout(createPlot, expectedUpdateInterval * 1000);
                    updateTimerStart = Date.now();
                }
            } else {
                loadFail();
            }
        }).fail(loadFail)
        .always(function () {
            ajaxPending = false;
        });
    };
    const refreshUpdateTimer = function () {
        if (ajaxPending)
            updateTimerEl.text('requesting data, please wait');
        else if (!autoRefresh)
            updateTimerEl.text('auto-refresh is disabled, refresh page to re-enable');
        else if (expectedUpdateInterval != null) {
            let remainTime = expectedUpdateInterval - (Date.now() - updateTimerStart) / 1000;
            if (remainTime < 0)
                updateTimerEl.text('updating');
            else
                updateTimerEl.text('next update expected in ' + formatTimer(remainTime));
        } else {
            updateTimerEl.text('no new data is being received, try refreshing the page');
        }
    };
    let relayoutLoadTimer = null;
    plotEl.on('plotly_relayout', function (ev, changes) {
        let rangeStart = null;
        let rangeEnd = null;
        if ('xaxis.range' in changes) {
            rangeStart = changes['xaxis.range'][0];
            rangeEnd = changes['xaxis.range'][1];
        } else if ('xaxis.range[0]' in changes) {
            rangeStart = changes['xaxis.range[0]'];
        } else if ('xaxis.range[1]' in changes) {
            rangeEnd = changes['xaxis.range[1]'];
        }
        adjustTicks();
        if (rangeStart != null) {
            rangeStart = (new Date(rangeStart)).getTime();
            if (rangeStart < loadedFrom) {
                if (relayoutLoadTimer !== null)
                    window.clearTimeout(relayoutLoadTimer);
                relayoutLoadTimer = window.setTimeout(function () {
                    const roundingMs = 6 * 60 * 60 * 1000;
                    rangeStart = Math.floor(rangeStart / roundingMs) * roundingMs;
                    createPlot(rangeStart, loadedFrom, true, 'prepend');
                    relayoutLoadTimer = null;
                    disableAutoRefresh();
                }, 500);
            }
        }
        // TODO: range end?
    });
    window.setInterval(refreshUpdateTimer, 1000);
    createPlot();
});
