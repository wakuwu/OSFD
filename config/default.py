base_attack = dict(
    IFGSM=dict(
        type="IFGSM",
        alpha=1.0
    ),
    MI=dict(
        type="MI",
        alpha=1.0,
        momentum=1.0
    ),
    DI=dict(
        type="DI",
        alpha=1.0,
        prob=1.0,
        scale=1.1
    ),
    RRB=dict(
        type="RRB",
        alpha=1.0,
        prob=1.0,
        theta=7.,
        l_s=10,
        rho=0.8,
        s_max=1.1,
        sigma=6.0
    )
)

transfer_attack = dict(
    OSFD=dict(
        type="OSFD",
        k=3.0
    )
)
