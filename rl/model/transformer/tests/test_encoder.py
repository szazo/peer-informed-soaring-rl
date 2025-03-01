from model.transformer.encoder import Encoder


def test_create_should_not_smoke():

    # when
    enc = Encoder(input_dim=5,
                  attention_internal_dim=6,
                  ffnn_hidden_dim=7,
                  ffnn_dropout_rate=0.1,
                  attention_head_num=3,
                  layer_count=3,
                  enable_layer_normalization=False)

    # then
    assert enc is not None
