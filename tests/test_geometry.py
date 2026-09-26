from estacionamiento.geometry import (
    EGRESO,
    INGRESO,
    NEGATIVE_TO_POSITIVE,
    LineCrossingCounter,
    box_center,
    point_on_segment,
    side_of_line,
)

LINE_START = (0, 100)
LINE_END = (200, 100)


def test_side_of_line_separa_los_dos_semiplanos():
    assert side_of_line((100, 150), LINE_START, LINE_END) > 0
    assert side_of_line((100, 50), LINE_START, LINE_END) < 0


def test_point_on_segment_acepta_cerca_y_rechaza_lejos():
    assert point_on_segment((100, 101), LINE_START, LINE_END, eps=2)
    assert not point_on_segment((100, 130), LINE_START, LINE_END, eps=2)
    assert not point_on_segment((400, 100), LINE_START, LINE_END, eps=2)


def test_box_center_es_el_punto_medio():
    assert box_center(10, 20, 30, 60) == (20, 40)


def test_cruce_de_positivo_a_negativo_cuenta_un_ingreso():
    counter = LineCrossingCounter(LINE_START, LINE_END, dist_thresh=18)
    assert counter.update(1, (100, 110)) is None
    event = counter.update(1, (100, 90))
    assert event is not None
    assert event.kind == INGRESO
    assert event.delta == -1
    assert counter.ingresos == 1
    assert counter.update(1, (100, 150)) is None
    assert counter.ingresos == 1


def test_cruce_inverso_cuenta_un_egreso():
    counter = LineCrossingCounter(LINE_START, LINE_END, dist_thresh=18)
    assert counter.update(2, (100, 90)) is None
    event = counter.update(2, (100, 110))
    assert event.kind == EGRESO
    assert event.delta == 1
    assert counter.egresos == 1


def test_el_sentido_de_ingreso_es_configurable():
    counter = LineCrossingCounter(
        LINE_START,
        LINE_END,
        dist_thresh=18,
        entry_direction=NEGATIVE_TO_POSITIVE,
    )
    counter.update(3, (100, 90))
    event = counter.update(3, (100, 110))
    assert event.kind == INGRESO
    assert event.delta == -1


def test_un_cruce_lejos_del_segmento_no_cuenta():
    counter = LineCrossingCounter(LINE_START, LINE_END, dist_thresh=18)
    counter.update(4, (100, 40))
    assert counter.update(4, (100, 160)) is None
    assert counter.ingresos == 0
    assert counter.egresos == 0
