from estacionamiento.spaces import count_space_labels


def test_cuenta_libres_y_ocupados_con_los_nombres_de_pklot():
    counts = count_space_labels(
        ["empty", "space-occupied", "occupied", "space-empty", "persona"],
        empty_names=["empty", "space-empty"],
        occupied_names=["occupied", "space-occupied"],
    )
    assert counts == {"empty": 2, "occupied": 2}
