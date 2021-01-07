$(window).on('load', function () {
    let input_field = $('#display');
    let accepts_input = true;
    let pending_ajax = false;
    let set_field = function (value) {
        input_field.val(value);
    };
    let reset_field = function () {
        input_field.data('actual-value', '')
        set_field('');
    };
    let flash_message = function (messages, on_done, flashes, on_time, off_time, repeats) {
        if (typeof(messages) == 'string')
            messages = [messages];
        if (flashes === undefined)
            flashes = (messages.length > 1) ? 0 : 3;
        if (on_time === undefined)
            on_time = 400;
        if (off_time === undefined)
            off_time = 250;
        if (repeats === undefined)
            repeats = 1;

        let set_msg_step = function (msg) {
            return function (next) {
                set_field(msg);
                next();
            };
        };
        let flash_on = function (next) {
            input_field.css('opacity', '1.0');
            next();
        };
        let flash_off = function (next) {
            input_field.css('opacity', '0.0');
            next();
        };
        let finish_step = function (next) {
            if (on_done !== undefined)
                on_done()
            accepts_input = true;
            next()
        };

        let qtarget = input_field;
        accepts_input = false;
        for (let r = 0; r < repeats; r++) {
            for (const msg of messages) {
                for (let f = 0; f < Math.max(flashes, 1); f++) {
                    qtarget.queue(set_msg_step(msg))
                           .queue(flash_on)
                           .delay(on_time);
                    if (flashes) {
                        qtarget.queue(flash_off)
                               .delay(off_time)
                               .queue(flash_on);
                    }
                }
            }
            if (repeats && !flashes) {
                qtarget.queue(flash_off)
                       .delay(off_time)
                       .queue(flash_on);
            }
        }
        qtarget.queue(finish_step);
    };
    let flash_error = function (messages, on_done, flashes, on_time, off_time, repeats) {
        input_field.addClass('error');
        let actual_on_done = function () {
            input_field.removeClass('error');
            on_done()
        };
        flash_message(messages, actual_on_done, flashes, on_time, off_time, repeats);
    }
    let type_digit = function (digit) {
        if (!accepts_input || pending_ajax)
            return;
        if (digit == undefined)
            return;
        input_field.data('actual-value', input_field.data('actual-value') + digit)
        input_field.val(input_field.val() + '*');
        if (input_field.data('actual-value').length > 6)
            flash_error('ERROR', reset_field);
    };
    let clear_pad = function () {
        if (pending_ajax)
            return;
        reset_field();
    };
    let submit = function () {
        if (pending_ajax)
            return;
        let secret_code = input_field.data('actual-value');
        pending_ajax = true;
        set_field('-WAIT-');
        $.ajax({
            url: '/auth',
            method: 'POST',
            cache: false,
            data: {secret_code: secret_code},
            dataType: 'json'
        }).done(function (data, textStatus, jqXHR) {
            if (data != null && data['status'] == 'SUCCESS') {
                Cookies.remove(data['session_cookie']);
                Cookies.set(data['session_cookie'], data['session_id'], { sameSite: 'strict' });
                flash_message(['CODE', 'VALID'], function () { window.location = "/"; }, 1, 750, undefined, 1);
            } else {
                flash_error(['ACCESS', 'DENIED'], reset_field, 0, 650, undefined, 2);
            }
        }).fail(function () {
            flash_error('FAILED', reset_field);
        }).always(function () {
            pending_ajax = false;
        });
    };
    $('button.nr_btn').on('click', function () { type_digit($(this).data('value')); });
    $('button#btn_cancel').on('click', clear_pad);
    $('button#btn_ok').on('click', submit);
    $(window).on('keydown', function (event) {
        let key = event.key;
        if (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].indexOf(key) != -1) {
            event.preventDefault();
            type_digit(key);
        } else if (key == 'Enter') {
            event.preventDefault();
            submit();
        } else if (key == 'Backspace' || key == 'Delete') {
            event.preventDefault();
            clear_pad();
        }
    });

    reset_field();
});