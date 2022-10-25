/**
 * JS for prediction.
 */
'use strict';

(function () {

    const languageSwitcher = document.querySelector('.language-switcher');
    const predictBtns = document.querySelectorAll('.predict-btn');
    const scenarioChoosers = document.querySelectorAll('.scenarios-choice');
    const startInput = document.getElementById('start-pollutant-date');
    const daysChoice = document.getElementById('days-choice-1');

    /*================================================================*/
    /* Key dates */

    const keyDates = {}
    const keyDatesInputs = document.querySelectorAll('.key-dates-values input')
    keyDatesInputs.forEach(function(input) {
        const datetimeStr = input.value.slice(0, 10);
        keyDates[input.id] = datetimeStr;
    });

    /*================================================================*/
    /* Language */

    // Get the user language.
    let lang = navigator.language;

    if ( ! lang || 'fr' != lang.substring(0, 2) ) {
        lang = 'en';
    } else {
        lang = 'fr';
    }

    setLanguage();

    // Set language.
    function setLanguage() {
        const classes = document.querySelector('body').classList;
        if (classes.contains('lang--' + lang)) {
            return false;
        }
        const removeLang = 'en' == lang ? 'fr' : 'en';
        classes.add('lang--' + lang);
        classes.remove('lang--' + removeLang);
        document.querySelector('html').setAttribute('lang', lang);
    }

    // Add language switcher event listeners.
    languageSwitcher.addEventListener('click', function(e) {
        e.preventDefault();
        if ( e.target.closest('a').classList.contains('lang--en') ) {
            lang = 'fr';
        } else {
            lang = 'en';
        }
        setLanguage();
    });

    /*================================================================*/
    /* Sticky menu */

    const options = {
        rootMargin: '180px',
        threshold: 1.0
    }

    const menuObserver = new IntersectionObserver(styleMenu, options);
    const menuContainer = document.querySelector('.cd-slider-nav');
    menuObserver.observe(menuContainer);
    const stuckClass = 'stuck';

    function styleMenu(entries) {
        if (menuContainer) {
            if (entries[0].isIntersecting) {
                menuContainer.classList.remove(stuckClass)
            } else {
                menuContainer.classList.add(stuckClass)
            }
        }
    }


    /*================================================================*/
    /* Prediction inputs */

    function getInputs(predictionContainer) {
        return predictionContainer.querySelectorAll(
            ':scope .inputs input[type="text"], :scope .inputs select');
    }

    function checkInputs(predictionContainer) {

        const highlights = []

        const inputs = getInputs(predictionContainer);

        inputs.forEach(function(input) {
            if (input != null) {
                if (!input.value) {
                    highlights.push(input)
                }
            }
        });

        clearHighlights(predictionContainer);

        if (highlights.length) {
            addHighlights(highlights);
            return false;
        }

        return true;

    }

    function addHighlights(highlights) {
        highlights.forEach(function(elem) {
            elem.classList.add('error');
        });
    }

    function clearHighlights(predictionContainer) {
        const highlighted = predictionContainer.querySelectorAll(':scope .error');
        highlighted.forEach(function(elem) {
            elem.classList.remove('error');
        });
    }

    predictBtns.forEach(function(predictBtn) {
        predictBtn.addEventListener( 'click', function(e) {

            e.preventDefault();

            const predictionContainer = e.target.closest('.prediction')

            const fieldsOK = checkInputs(predictionContainer)

            if (!fieldsOK) {
                return false;
            }

            const queryParams = getPredictionParams(predictionContainer)

            // Show loader.
            toggleLoader();

            // Remove focus from button.
            predictBtn.blur();

            const anchor = e.target.tagName.toLowerCase() == 'a' ? e.target : e.target.closest('a');

            makeGetRequest(anchor.href, queryParams)
                .then((data) => {
                    if ('error' in data) {
                        console.log(data.error)
                    } else {
                        updatePlotUrls(e.target, data)
                    }
                    toggleLoader()
                });

        });
    });

    const datepickers = document.querySelectorAll('.flatpickr');

    datepickers.forEach(function(datePicker) {

        const config = {
            'allowInput' : true
        };

        // Get any start and end dates/
        const datePickerContainer = datePicker.parentElement;
        const minInput = datePickerContainer.querySelector(':scope .min-date');
        const maxInput = datePickerContainer.querySelector(':scope .max-date');

        if (minInput) {
            config['minDate'] = minInput.value
        }

        if (maxInput) {
            config['maxDate'] = maxInput.value
        }

        flatpickr(datePicker, config)

    });

    /*================================================================*/
    /* Scenarios */

    // On change of scenario, change the other language dropdown
    // and set the date and days accordingly.
    scenarioChoosers.forEach(function(scenarioChooser) {

        scenarioChooser.addEventListener( 'change', function(e) {
            matchScenarioChooserValues(e.target)
            runScenario(e.target.value);
        });

    });

    // On change of date or days, reset the scenario choosers.
    startInput.addEventListener('change', resetScenarios);
    daysChoice.addEventListener('change', resetScenarios);

    function matchScenarioChooserValues(activeChooser) {
        const inactiveChooserLang = activeChooser.id == 'scenarios-choice-en' ? 'fr' : 'en';
        const inactiveChooser = document.getElementById(`scenarios-choice-${inactiveChooserLang}`);
        inactiveChooser.value = activeChooser.value;
    }

    function runScenario(scenario) {
        const scenarioData = {
            'forecast-3' : ['last_forecast_start', 3],
            'last-training' : ['last_5_training', 5],
            'last-validation' : ['last_5_validation', 5],
            'last-known' : ['last_3_actual', 3]
        }

        const fp = document.querySelector("#start-pollutant-date")._flatpickr;
        fp.setDate(keyDates[scenarioData[scenario][0]]);

        // startInput.value = keyDates[scenarioData[scenario][0]]
        // // If we have a mobile input then we need to change the value of that too.
        // const startInputSibling = startInput.nextElementSibling
        // if (startInputSibling && startInputSibling.classList.contains('flatpickr-mobile')) {
        //     startInputSibling.value = keyDates[scenarioData[scenario][0]];
        // }
        daysChoice.value = scenarioData[scenario][1]

    }
    function resetScenarios() {
        scenarioChoosers.forEach(function(chooser) {
            chooser.value = '';
        });
    }

    /*================================================================*/
    /* Pollutant plots */

    function updatePlotUrls(target, data) {
        const predictionContainer = target.closest('.prediction')
        const imgPlots = predictionContainer.querySelectorAll(':scope .plot')
        imgPlots.forEach(function(imgPlot) {
            if (imgPlot.classList.contains('lang--en')) {
                if (imgPlot.classList.contains('plot--narrow')) {
                    imgPlot.src = `${data.plots_dir}/en-narrow.png`
                } else {
                    imgPlot.src = `${data.plots_dir}/en-wide.png`
                }
            } else {
                if (imgPlot.classList.contains('plot--narrow')) {
                    imgPlot.src = `${data.plots_dir}/fr-narrow.png`
                } else {
                    imgPlot.src = `${data.plots_dir}/fr-wide.png`
                }
            }
        });
    }

    function getPredictionParams(predictionContainer) {

        let queryParams = {}

        const startInput = predictionContainer.querySelector(':scope .start-date');
        const daysInput = predictionContainer.querySelector(':scope .days-choice');
        const pollutantDropdown = predictionContainer.querySelector(':scope .pollutant-choice')

        if (startInput) {
            queryParams['start'] = startInput.value
        }

        if (daysInput) {
                queryParams['days'] = daysInput.value
        }

        if (pollutantDropdown) {
            queryParams['pollutant'] = pollutantDropdown.value
        }

        return queryParams

    }

    async function makeGetRequest(url = '', queryParams = {}) {

        let urlQueryParams = [];
        for (const [key, value] of Object.entries(queryParams)) {
            urlQueryParams.push(`${key}=${value}`);
        }

        urlQueryParams = urlQueryParams.join('&');

        url = `${url}?${urlQueryParams}`;

        // Default options are marked with *
        const response = await fetch(url, {
            method: 'GET', // *GET, POST, PUT, DELETE, etc.
            mode: 'cors', // no-cors, *cors, same-origin
            cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
            credentials: 'same-origin', // include, *same-origin, omit
            redirect: 'follow', // manual, *follow, error
            referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
        });

        return response.json(); // parses JSON response into native JavaScript objects

    }

    function toggleLoader() {
        const loader = document.querySelector('.loading-overlay');
        if (loader) {
            loader.classList.toggle('active');
        }
    }

    /*================================================================*/
    /* Custom links */

    const performanceMenuItem = document.getElementById('performance-menu-item');

    const performancePollutantLinks = document.querySelectorAll('.performance-pollutant-link');
    performancePollutantLinks.forEach(function(performancePollutantLink) {
        performancePollutantLink.addEventListener('click', function(e) {
            e.preventDefault();
            console.log(1);
            performanceMenuItem.click();
            setTimeout(function() {
                document.getElementById('performance-pollutant').scrollIntoView();
            }, 500)

        });
    });

})();