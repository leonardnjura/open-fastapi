import os
import traceback
import re
from dotenv import load_dotenv
from typing import Any, Container, Iterable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from instaloader import Instaloader, Profile, Post
from itertools import islice
import urllib.parse
import urllib.request
import google.generativeai as genai
import PIL.Image
from io import BytesIO
import base64
from g4f.client import Client

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
X_TOKEN = os.getenv('X_TOKEN')


app = FastAPI()

# REFs
# https://youtube.com/watch?v=8R-cetf_sZ4
# https://ai.google.dev/gemini-api
# https://fastapi.tiangolo.com/tutorial/cors/

origins = [
    "http://localhost:3000",
    "https://www.countryooze.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

L = Instaloader()

gptClient = Client()


def ig_probe() -> str:
    return ""


def download_one_ig_post_by_shortcode(shortcode: str):
    # https://github.com/instaloader/instaloader/blob/0e13c73b6296302bee854357664000f36fb7cf46/test/instaloader_unittests.py#L67
    # https://instaloader.github.io/cli-options.html
    # https://www.instagram.com/p/SHORTCODE/
    # NOTE:: No login required for publicly available items

    # L.load_session_from_file(f"dnyaberry")
    SHORTCODE = shortcode

    post = Post.from_shortcode(L.context, SHORTCODE)

    url = L.get_post_media(post)

    encoded_url = urllib.parse.quote(url, safe=" ")

    print(f'!!instaloader {encoded_url} ----> ??')
    print(f'!!instaloader {SHORTCODE} ----> ??')

    print(f'\n\n')
    mediaid = post.mediaid
    caption = post.caption
    date_utc = post.date_utc
    is_video = post.is_video
    url = post.url  # photo url / video cover
    video_url = post.video_url  # if video exists
    owner = post.owner_profile.username
    owner_full_name = post.owner_profile.full_name

    lno_post = {
        "mediaid": mediaid,
        "caption": caption,
        "date_utc": date_utc,
        "url": url,
        "video_url": video_url,
        "is_video": is_video,
        "owner": f'@{owner}',
        "owner_full_name": owner_full_name
    }

    return lno_post


def download_recent_ig_posts(profile_of_interest: str, limit=5):
    # https://github.com/instaloader/instaloader/blob/0e13c73b6296302bee854357664000f36fb7cf46/test/instaloader_unittests.py#L67
    # https://instaloader.github.io/cli-options.html
    # NOTE:: No login required for publicly available items

    PROFILE_OF_INTEREST = profile_of_interest

    profile = Profile.from_username(L.context, PROFILE_OF_INTEREST)
    posts = profile.get_posts()
    LIMIT = limit

    print(f'!!instaloader @{PROFILE_OF_INTEREST} ----> PUBLIC')

    print(f'\n\n')
    recent_posts = []
    for post in islice(posts, LIMIT):
        mediaid = post.mediaid
        caption = post.caption
        date_utc = post.date_utc
        is_video = post.is_video
        url = post.url  # photo url / video cover
        video_url = post.video_url  # if video exists
        owner = post.owner_profile.username
        owner_full_name = post.owner_profile.full_name

        lno_post_entry = {
            "mediaid": mediaid,
            "caption": caption,
            "date_utc": date_utc,
            "url": url,
            "video_url": video_url,
            "is_video": is_video,
            "owner": f'@{owner}',
            "owner_full_name": owner_full_name
        }

        recent_posts.append(lno_post_entry)

        print(f'!!instaloader post ----> {lno_post_entry}')
        print(f'\n')

    ig_rpt = {
        "profile": f'@{PROFILE_OF_INTEREST}',
        "recent_posts": recent_posts,
        "limit": LIMIT
    }
    return ig_rpt


def download_saved_ig_posts_for_user(username: str, limit=5):
    # https://github.com/instaloader/instaloader/blob/0e13c73b6296302bee854357664000f36fb7cf46/test/instaloader_unittests.py#L67
    # https://instaloader.github.io/cli-options.html
    # NOTE:: Login required in terminal ==> $ instaloader --login <username>

    a = username
    username = a[1:] if a.startswith('@') else a

    # Login or load session
    # L.login(username, password)
    L.load_session_from_file(f"{username}")
    PROFILE_OF_INTEREST = username

    profile = Profile.from_username(L.context, PROFILE_OF_INTEREST)
    followers = profile.get_followers()  # Login required
    followees = profile.get_followees()  # Login required
    saved_posts = profile.get_saved_posts()  # Login as owner
    LIMIT = limit

    followers_count = followers.count
    followees_count = followees.count
    print(
        f'!!instaloader @{PROFILE_OF_INTEREST} ----> followers {followers_count} | following {followees_count}')

    print(f'\n\n')
    recent_saved_posts = []
    for post in islice(saved_posts, LIMIT):
        mediaid = post.mediaid
        caption = post.caption
        date_utc = post.date_utc
        is_video = post.is_video
        url = post.url  # photo url / video cover
        video_url = post.video_url  # if video exists
        owner = post.owner_profile.username
        owner_full_name = post.owner_profile.full_name

        lno_post_entry = {
            "mediaid": mediaid,
            "caption": caption,
            "date_utc": date_utc,
            "url": url,
            "video_url": video_url,
            "is_video": is_video,
            "owner": f'@{owner}',
            "owner_full_name": owner_full_name
        }

        recent_saved_posts.append(lno_post_entry)

        print(f'!!instaloader saved post ----> {lno_post_entry}')
        print(f'\n')

    ig_rpt = {
        "profile": f'@{PROFILE_OF_INTEREST}',
        "followers": followers_count,
        "following": followees_count,
        "recent_saved_posts": recent_saved_posts,
        "limit": LIMIT
    }
    return ig_rpt


def extract_image_urls(s: str):
    result_list: list[str] = []
    pat = re.compile(
        r'(?i)https?[^<>\s\'\"=]+(?:jpg|png|webp)\b|[^:<>\s\'\"=]+(?:jpg|png|webp)\b')
    for m in pat.findall(s):
        # print(f'!!image url ----> {m}')
        result_list.append(m)
    return result_list


def chat(system_instructions, conversation_context, last_message, model):
    # https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
    # conversation_context = previous_conversation + [user_msg]

    # NOTE gpt-3.5-turbo cannot view images directly but we still can have fun with embeded urls as we wait for gpt-4o ))
    detected_image_urls = extract_image_urls(last_message)
    is_image_detected_in_prompt = len(detected_image_urls) > 0
    print(f'!!last_message ----> {last_message}')
    print(f'!!detected_image_urls ----> {detected_image_urls}')

    assert isinstance(system_instructions, str), "`system` should be a string"
    assert isinstance(
        conversation_context, list), "`conversation_context` should be a list"
    system_msg = [{"role": "system", "content": system_instructions}]
    user_assistant_msgs = [
        {"role": "assistant", "content": conversation_context[i]} if i % 2 else {
            "role": "user", "content": conversation_context[i] if is_image_detected_in_prompt == False else [{"type": "text", "text": conversation_context[i]},
                                                                                                             {
                "type": "image_url",
                "image_url": {
                    "url": detected_image_urls[0],
                },
            },]}
        for i in range(len(conversation_context))]

    msgs = system_msg + user_assistant_msgs
    response = gptClient.chat.completions.create(model=model,
                                                 messages=msgs)
    finish_reason = response.choices[0].finish_reason
    assert finish_reason == "stop", f"The status code was {finish_reason}."
    # response..
    prediction = response.choices[0].message.content
    # update conversation context list/arr, :/
    conversation_context_updated = conversation_context
    if prediction is not None:
        conversation_context_updated = conversation_context + [prediction]
    return prediction, conversation_context_updated, finish_reason


async def generate_image(prompt, model):
    # este es f_cked de verdad o no sabemos?
    response = gptClient.images.generate(
        model=model,
        prompt=prompt,
    )

    image_url = response.data[0].url
    b64_json = response.data[0].b64_json
    return image_url, b64_json


class IgQueryProtectedProfile(BaseModel):
    protectedProfileUsername: str
    limit: int = None


class IgQueryPublicProfile(BaseModel):
    publicProfileUsername: str
    limit: int = None


class Item(BaseModel):
    id: str  # don't start with _underscore!!
    task: str
    allocatedDays: int = None
    isDone: bool = False

    def update(
        self,
        other: Iterable[tuple[str, Any]],
        exclude: Container[str] = (),
    ) -> None:
        for field_name, value in self:
            setattr(self, field_name, value)
        for field_name, value in other:
            print('!!field_name' + field_name)
            if field_name not in exclude and value != None:
                print('!!may update ------> ' +
                      field_name + ' | ' + str(value))
                setattr(self, field_name, value)


class ItemUpdate(BaseModel):
    task: str = None
    allocated_days: int = None
    isDone: bool = None


class ImageChatGemini(BaseModel):
    commandPrompt: str
    base64Data: str
    geminiModelId: str = None


class TextChatGpt(BaseModel):
    commandPrompt: str
    gptModelId: str = None
    conversationContext: list[str]
    isTravelMode: bool = None


class ImageGenGemini(BaseModel):
    commandPrompt: str
    geminiModelId: str = None


items: list[Item] = []


@app.get("/")
def root(request: Request):
    my_header = request.headers.get('my-header')
    try:
        return {
            "Hello": "X",
        }
    except:
        traceback.print_exc()
        msg = "Server says urgh----"
        raise HTTPException(status_code=500, detail={"msg": msg})


@app.post("/api/items", response_model=Item, status_code=201)
def create_item(item: Item):
    items.append(item)
    return item


@app.post("/api/ai/image-chat")
def chat_about_image_with_gemini(request: Request, item: ImageChatGemini):
    # gemini api key placed in here as endpoint is public
    # pass yours in header as my-google-gemini-api-key
    # TODO jwt / cookies auth
    # GOOGLE_GEMINI_API_KEY = os.getenv('GOOGLE_GEMINI_API_KEY')
    my_google_gemini_api_key = request.headers.get('my-google-gemini-api-key')
    default_model_id = 'gemini-1.0-pro-latest'
    genai.configure(api_key=my_google_gemini_api_key)

    try:

        model_id = default_model_id
        if item.geminiModelId is not None:
            model_id = item.geminiModelId

        model = genai.GenerativeModel(model_id)
        command_prompt = item.commandPrompt
        base64_data_raw = item.base64Data

        # data:image/jpeg;base64,/9j~
        base64_data_popped = base64_data_raw.split(";base64,", 1)[1]

        # img = PIL.Image.open('/Users/leo/Downloads/007.jpg')
        img = PIL.Image.open(BytesIO(base64.b64decode(base64_data_popped)))
        # response = model.generate_content(command_prompt)
        # NOTE use a model that supports img or uncomment above
        response = model.generate_content([command_prompt, img])

        prediction = response.text

        return {
            "message": "all fields",
            "data": {
                "modelId": model_id,
                "prediction": prediction,
                "predictionType": "gemini",
                "prompt": command_prompt,
                "conversationId": "xxxxxx="
            },
            "verbose": {
                "mode": "base64"
            }
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh----"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


@app.post("/api/ai/image-gen")
async def image_gen(request: Request, item: ImageGenGemini):
    # TODO jwt / cookies auth
    default_model_id = 'gemini'

    try:

        model_id = default_model_id
        if item.geminiModelId is not None:
            model_id = item.geminiModelId

        command_prompt = item.commandPrompt

        image_url, b64_json = await generate_image(command_prompt, model_id)

        return {
            "message": "all fields",
            "data": {
                "modelId": model_id,
                "imageUrl": image_url,
                "b64Json": b64_json,
                "predictionType": "gemini",
                "prompt": command_prompt,
                "conversationId": "xxxxxx="
            },
            "verbose": {
                "mode": "prompt"
            }
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh----"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


@app.post("/api/ai/text-chat")
def chat_about_text(request: Request, item: TextChatGpt):
    # we will dedicate this endpoint to text chatting
    # ***********************************************
    # gpt-3.5-turbo is the preferred default but doesn't support images
    # gpt-4o supports images but takes 10+ seconds and throws SSLCertVerificationError..
    # ..[may be a good price to pay but bad news for users and bad news for me on vercel for Hobby tier]
    # https://community.openai.com/t/gpt-4o-tokens-per-second-comparable-to-gpt-3-5-turbo-data-and-analysis/768559
    # https://neuroflash.com/blog/gpt-3-wiki/
    # https://github.com/xtekky/gpt4free?tab=readme-ov-file
    # TODO keep checking if gpt-4 certificate has cleared or ill bgates is dead && unleash image-chat with gpt-4o, or get rich like melon usk :/
    # TODO jwt / cookies auth
    default_model_id = 'gpt-3.5-turbo'

    try:
        model_id = default_model_id
        if item.gptModelId is not None:
            model_id = item.gptModelId

        command_prompt = item.commandPrompt

        # NOTE role types------
        # "system" messages describe the behavior of the AI assistant.
        # "user" messages describe what you want the AI assistant to say.
        # "assistant" messages describe previous responses in the conversation.

        # Define the system message
        system_msg_default = 'You are a helpful assistant who understands everything. Respond only in English'
        system_msg_countryooze = '''
        ## Context
        You are a travel assistant bot for Countryooze LLC that has a website at https://www.countryooze.com. Provide precise and concise facts about country facts, feed and travel preparedness in a friendly and informative manner. Also  for any country allow queries on current weather, current time in capital, education/curriculum format, electricity plug/socket types, heads of state/government, travel advisory and recent Al Jazeera/France 24 news. Return the main country page as the link to any of these.

        ## Rules for the responses:
        ### When citing sources match the the ISO 3166-1 alpha-2 code for a country named with the urls in the Knowledge base:
        - For main country page return https://countryooze.com/c/[COUNTRY_CODE]
        - For administrative divisions in a country return https://countryooze.com/divisions/[COUNTRY_CODE]
        - For travel tips return https://countryooze.com/travel-tips
        - For getting prepared tips return https://www.countryooze.com/getting-prepared
        - For tips to avoid tourist traps return https://www.countryooze.com/tourist-traps
        - For tips to avoid tourist scams return https://www.countryooze.com/tourist-scams
        - For per country tips to avoid tourist don'ts return https://www.countryooze.com/the-donts/[COUNTRY_CODE]
        - For timezones or current time return https://countryooze.com/times-in-capital/[COUNTRY_CODE]
        - For national anthems return https://countryooze.com/national-anthem/[COUNTRY_CODE]
        - For tourist attractions return https://countryooze.com/tourist-attractions/[COUNTRY_CODE]
        - For per country tourist attractions saved in the user's "MUST SEE LIST" a.k.a "MUST LIST" return https://countryooze.com/must-list/[COUNTRY_CODE]
        - For national holidays return https://countryooze.com/national-holidays/[COUNTRY_CODE]
        - For visas return https://countryooze.com/visa/[COUNTRY_CODE]
        - For AI flag chat return https://countryooze.com/image-chat/flag/[COUNTRY_CODE]
        - For AI image chat return https://countryooze.com/image-chat
        - For AI image chat for Android return https://play.google.com/store/apps/details?id=com.codesandme.ai_image_chat

        ### When citing sources for national languages spoken in a country, match the the ISO 639-3 code for a language named with the urls in the Knowledge base:
        - For languages return https://countryooze.com/phrasebook/[LANGUAGE_CODE]


        ### When citing sources for currency used in a country or forex, lookup the iso ISO 4217 code for a main currency used in the named country or countries with the urls in the Knowledge base:
        - For currencies return https://countryooze.com/currency/[CURRENCY_CODE]


        #### Examples:
        - If the prompt or response has a country named United Arab Emirates, then COUNTRY_CODE is "AE" i.e. the iso 3166-1 alpha-2 code for United Arab Emirates

        - If the prompt or response has a language named Spanish, then LANGUAGE_CODE is "spa" i.e. the iso 639-3 code for Spanish

        - If the prompt or response has a currency named Philippine Peso or a country named Philippines, then CURRENCY_CODE is "PHP" i.e. the iso 4217 code for the Philippine Peso

        - If the prompt or response has a currency named Euro or a country that uses the Euro like France, Germany, Luxembourg, etc, then CURRENCY_CODE is "EUR" i.e. the iso 4217 code for the Euro

        ### Use markdown markup language to include clickable and user-friendly links
        - For example, if the Image Chat url is https://countryooze.com/image-chat then return the markdown [Image Chat](https://countryooze.com/image-chat)
        - Likewise for the Countryooze Chrome browser extension link return [Chrome Extension](https://chrome.google.com/webstore/detail/countryooze-extension/gpmljcfhmfinnamkaenonfemmbbecgmc) and not the exposed url.
        - Likewise for the GPT-3 Countryooze bot return [GPT-3 Assistant](https://countryooze.com/chat) 
        - Likewise for the GPT-4o Countryooze bot return [GPT-4o Assistant with Image Support](https://poe.com/BotCountryoozeGPT-4o)
        
        - Respond in English
        - If the response has a link containing "countryooze.com" don't say "our website" say "our page"
        - No responses involving competitors
        - Don't cite more than 2 sources


        ## Knowledge base:
        ### Knowledge base for our website links:
        - https://countryooze.com/sitemap.xml
        ### Knowledge base for our affiliate links for Tours & Airport Transfers:
        - https://wegotrip.tp.st/v8UdgDq3
        - https://bikesbooking.tp.st/V72s4adk
        - https://kiwitaxi.tp.st/bkyIPpYB
        - https://tp.st/eFikMsn8
        - https://wayaway.tp.st/2fs5WpWN
        - https://getrentacar.tp.st/OQvBoAYY
        ### Knowledge base for our affiliate links for Hotel Bookings and Accommodations:
        - https://hotellook.tp.st/WWnotjXA
        '''
        system_msg = system_msg_default
        if item.isTravelMode:
            system_msg = system_msg_countryooze

        # Define the [new] user message
        user_msg = command_prompt

        # TODO bring in previous conversation i.e. assistant response(s) + user prompt(s) if any
        previous_conversation = item.conversationContext

        # Define the assistant message(s) i.e. create an array of user and assistant messages
        assistant_msg = previous_conversation + [user_msg]

        # TAKE I-----------------------------------------------/
        # response = gptClient.chat.completions.create(
        #     model=model_id,
        #     messages=[{"role": "system", "content": system_msg},
        #               {"role": "user", "content": user_msg}],
        # )
        # prediction = response.choices[0].message.content
        # finish_reason = response.choices[0].finish_reason

        # TAKE II----------------------------------------------/
        prediction, conversation_context_updated, finish_reason = chat(
            system_msg, assistant_msg, user_msg, model_id)

        return {
            "message": "all fields",
            "data": {
                "modelId": model_id,
                "prediction": prediction,
                "predictionType": "gpt",
                "prompt": command_prompt,
                "conversationId": "xxxxxx=",
                "conversationContext": conversation_context_updated,
            },
            "verbose": {
                "mode": "base64",
                "finish_reason": finish_reason,
            }
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh----"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


@app.post("/api/ig/posts")
def get_ig_posts(item: IgQueryPublicProfile):
    try:
        a = item.publicProfileUsername
        profile_username = a[1:] if a.startswith('@') else a
        limit = item.limit

        rpt = download_recent_ig_posts(profile_username, limit)
        return {
            "data": rpt,
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh----"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


@app.post("/api/ig/savedposts")
def get_ig_posts_protected(item: IgQueryProtectedProfile):
    try:
        a = item.protectedProfileUsername
        profile_username = a[1:] if a.startswith('@') else a
        limit = item.limit
        rpt = download_saved_ig_posts_for_user(profile_username, limit)
        return {
            "data": rpt,
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh---- | $ instaloader --login <username>"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


@app.get("/api/items/{id}")
def get_one(id: str):
    for d in items:
        if d.id == id:
            print("!!found a match for get:: "+id)
            return d
    else:
        # adDnote:: avoid response model here as 404 body is not compliant ))
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/api/ig/posts/{shortcode}")
def get_one_post(shortcode: str):
    try:
        rpt = download_one_ig_post_by_shortcode(shortcode)
        return {
            "data": rpt,
        }
    except Exception as e:
        traceback.print_exc()
        msg = "Server says urgh----"
        verbose = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={
                            "msg": msg, "verbose": verbose})


# adDnote:: avoid response model here as 404 body is not compliant ))
# we actually perform a partial update [with PATCH]
@app.put("/api/items/{id}")
def update_one(id: str, partialUpdates: ItemUpdate):
    counter = 0
    for d in items:

        if d.id == id:
            print("!!found a match for put:: "+id)
            idx = counter

            obj1 = d
            obj2 = partialUpdates

            obj1.update(obj2, exclude={"id"})
            print(obj1)

            # HACK
            # pop old
            items.remove(d)

            # insert updated
            items.insert(idx, obj1)

            return obj1
        counter = counter+1
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/api/items", response_model=list[Item])
def get_all(limit: int = 10):
    if limit == 0:
        return items
    else:
        return items[0:limit]


@app.delete("/api/items/{id}", status_code=204)
def delete_one(id: str):
    for d in items:
        if d.id == id:
            print("!!found a match for chucky:: "+id)
            items.remove(d)
            return
    else:
        raise HTTPException(status_code=404, detail="Item not found")
