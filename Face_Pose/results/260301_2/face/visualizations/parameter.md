dropout : 0.3
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="权重衰减"
    )