## CocktailBert: KoBERT 기반 칵테일 추천 언어 모델

### About

- https://github.com/SKTBrain/KoBERT 에 공개된 모델을 바탕으로, 칵테일의 묘사 및 느낌에 관한 문장과 categorized information data를 활용하여 fine-tuning하여 사용자의 느낌을 기반으로 db 내 칵테일을 추천합니다.
- 해당 기능은 칵테일 커뮤니티 사이트 tipsynight.site의 **AI에게 추천받기**를 통해서도 접할 수 있습니다.
- Training Data: TODO

- 사용된 Categorization

  ```python
  size = {
    0: "long",
    1: "short",
    2: "shot"
  }

  ABV = {
      0: (0,15),
      1: (15,30),
      2: (30,100)
  }

  color = {
    0: "red",
    1: "green",
    2: "blue",
    3: "brown",
    4: "yellow"
  }
  ```

### Requirements

- see [requirements.txt](https://github.com/AlongwithKiman/cocktailbert/blob/main/requirements.txt)

### Get started

- Install python packages and download checkpoint

  - you may need to install gdown, or you can download it from [ckpt drive link](https://drive.google.com/file/d/1olPNiRHSs1qzyHxFb72cR4Sw8OX87dSW/view)

  ```sh
  pip install -r requirements.txt

  # downloading checkpoint
  pip install gdown
  sh download_ckpt.sh
  ```

## How to use

- Get recommendation without visualization

  ```sh
  python inference.py --sentence "자기 전 한 잔으로 취하고 싶어"

  ```

  **Result**
  | name | filter_type_two | ABV | color | image |
  |------------|-----------------|------|--------|----------------------------------------------------------------------------------------------------------|
  | 브롱크스 | short | 28.5 | f6c47c | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/brunch.jpg) |
  | BMW | short | 26.0 | e8a86d | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/bmw.jpg) |
  | 그래스호퍼 | short | 16.0 | 6fab70 | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/grasshopper.jpg) |
  | 브랜디 에그노그 | short | 15.7 | d5ac8f | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/brandy-eggnog.jpg) |
  | 네그로니 | short | 27.7 | db6340 | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/negroni.jpg) |
  | 모히또 | short | 18.9 | ffffff | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/mojito.jpg) |
  | 머드슬라이드 | short | 24.3 | 796d61 | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/mudslide.jpg) |
  | 화이트 러시안 | short | 22.5 | 76797e | [Image](https://qualla-image.s3.ap-northeast-2.amazonaws.com/white-russian.jpg) |

- Get recommendation **with** visualization

  - you should define font path on **--font_path** argument


  ```sh
  python3 inference.py --sentence "과일이 들어간 달달한 롱드링크가 먹고 싶어" --visualize --font_path "./fonts/NanumGothic.ttf"

  ```

  **Result**
  
  <img width="625" alt="스크린샷 2024-03-07 오전 12 29 46" src="https://github.com/AlongwithKiman/cocktailbert/assets/43671432/d21aa6f1-9b5a-464f-98eb-d3849f5ef31f">

## Fine-tuning

```sh
python train.py

```

## License

- TODO
